"""torch Datasets and loaders for all five sources, against the frozen contract.

The HDF5 cache is read **lazily and per worker**. Two reasons: h5py handles do
not survive a fork, so a handle opened in the parent silently corrupts reads in
workers; and MTAT's cache is ~750 MB, which -- multiplied by 4 workers on a
16 GB machine that is also holding a BERT -- is not a budget that survives
contact with reality. Each worker opens its own read-only handle on first use
and reads one track at a time.
"""
from __future__ import annotations

import json
import math
from itertools import cycle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph_builder import build_segment_graph
from .utils import REAL, get_generator, get_logger, resolve_path, seed_worker

LOGGER = get_logger("gbmc.data")

__all__ = [
    "MusicGraphDataset",
    "MelSpecDataset",
    "make_loader",
    "alternating_loader",
    "collate_texts",
    "load_tag_vocab",
]


def _read_manifest(manifest) -> pd.DataFrame:
    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(resolve_path(manifest))
    if not isinstance(manifest, pd.DataFrame):
        manifest = pd.DataFrame(manifest)
    frame = manifest.copy().reset_index(drop=True)
    if "y_tags" in frame.columns:
        frame["y_tags"] = frame["y_tags"].apply(_as_tag_list)
    return frame


def _as_tag_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        parsed = json.loads(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (TypeError, ValueError):
        return []


def load_tag_vocab(path) -> list[str]:
    """Read a tag vocabulary from JSON (list, or ``{"tags": [...]}``)."""
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return list(payload["tags"] if isinstance(payload, dict) else payload)


class MusicGraphDataset(Dataset):
    """Contract-compliant :class:`~torch_geometric.data.Data` per track.

    Three storage backends, chosen by what the caller passes:

    * ``graph_dir`` -- pre-built ``.pt`` graphs (synthetic data, sample graphs)
    * ``h5_path``   -- the feature cache; graphs are assembled on the fly
    * neither       -- features must already be inline in the manifest

    ``norm_stats`` must have come from the train split (see
    :func:`~src.audio_features.compute_norm_stats`); it is applied identically
    to val and test.
    """

    def __init__(
        self,
        manifest,
        cfg=None,
        h5_path=None,
        graph_dir=None,
        tag_vocab: Sequence[str] | None = None,
        norm_stats: dict | None = None,
        split: str | None = None,
        datasets: Sequence[str] | None = None,
    ):
        self.cfg = cfg
        self.manifest = _read_manifest(manifest)
        if split is not None and "split" in self.manifest.columns:
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if datasets is not None and "dataset" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["dataset"].isin(list(datasets))
            ].reset_index(drop=True)

        self.h5_path = str(resolve_path(h5_path)) if h5_path else None
        self.graph_dir = resolve_path(graph_dir) if graph_dir else None
        self.norm_stats = norm_stats
        self.tag_vocab = list(tag_vocab) if tag_vocab is not None else self._infer_vocab()
        self.tag_index = {tag: i for i, tag in enumerate(self.tag_vocab)}
        self._store = None          # opened lazily, per worker

    # -- vocabulary ------------------------------------------------------- #
    def _infer_vocab(self) -> list[str]:
        if "y_tags" not in self.manifest.columns:
            return []
        seen: list[str] = []
        pool: set[str] = set()
        for tags in self.manifest["y_tags"]:
            for tag in tags:
                if tag not in pool:
                    pool.add(tag)
                    seen.append(tag)
        return sorted(seen)

    @property
    def n_tags(self) -> int:
        return len(self.tag_vocab)

    def __len__(self) -> int:
        return len(self.manifest)

    # -- storage ---------------------------------------------------------- #
    def _handle(self):
        """Open this worker's own read-only HDF5 handle on first use."""
        if self._store is None and self.h5_path:
            import h5py

            self._store = h5py.File(self.h5_path, "r")
        return self._store

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = None      # never pickle an h5py handle to a worker
        return state

    def _load_features(self, track_id: str, row) -> np.ndarray:
        if "features" in row and isinstance(row["features"], np.ndarray):
            return row["features"]
        store = self._handle()
        if store is None or track_id not in store:
            raise KeyError(
                f"no cached features for {track_id!r}; run "
                "`python -m src.audio_features` / make features first"
            )
        # h5py slices straight off disk -- the cache is never fully resident
        return np.asarray(store[track_id][...], dtype=np.float32)

    # -- item ------------------------------------------------------------- #
    def _multi_hot(self, tags) -> np.ndarray | None:
        """Multi-hot vector, or ``None`` when this dataset has no tag labels.

        ``None`` becomes the ``-1`` sentinel row downstream. An empty tag list
        for a corpus that *does* carry tags is a genuine all-negatives row; a
        corpus with no tag vocabulary at all (DEAM, FMA) must not be scored as
        50 confident negatives.
        """
        if not self.tag_vocab:
            return None
        vector = np.zeros(len(self.tag_vocab), dtype=np.float32)
        hit = False
        for tag in tags:
            idx = self.tag_index.get(tag)
            if idx is not None:
                vector[idx] = 1.0
                hit = True
        return vector if (hit or tags is not None and len(tags) > 0) else None

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        track_id = str(row["track_id"])
        dataset_name = str(row.get("dataset", ""))

        if self.graph_dir is not None:
            path = self.graph_dir / f"{track_id}.pt"
            data = torch.load(path, weights_only=False)
            data.split = str(row.get("split", getattr(data, "split", "")))
            return data

        feats = self._load_features(track_id, row)
        if self.norm_stats is not None:
            from .audio_features import apply_norm

            feats = apply_norm(feats, self.norm_stats)

        tags = row["y_tags"] if "y_tags" in row else []
        has_tag_labels = dataset_name in {"mtat", "musiccaps"} or bool(tags)
        y_tags = self._multi_hot(tags) if has_tag_labels else None
        if y_tags is None and self.tag_vocab:
            y_tags = np.full(len(self.tag_vocab), -1.0, dtype=np.float32)

        return build_segment_graph(
            feats,
            self.cfg,
            y_tags=y_tags,
            n_tags=len(self.tag_vocab) or None,
            y_genre=_int_or_none(row.get("y_genre")),
            y_valence=_float_or_nan(row.get("y_valence")),
            y_arousal=_float_or_nan(row.get("y_arousal")),
            track_id=track_id,
            artist_id=str(row.get("artist_id", "")),
            dataset=dataset_name,
            provenance=str(row.get("provenance", REAL)),
            split=str(row.get("split", "")),
            text=str(row.get("text", "")),
        )


def _int_or_none(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return -1
        return int(value)
    except (TypeError, ValueError):
        return -1


def _float_or_nan(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class MelSpecDataset(Dataset):
    """Fixed-size log-mel patches for the CNN baseline (B2).

    Serves the *same* tracks and the *same* labels as the graph dataset, from a
    cache written by the same feature pipeline -- otherwise the CNN-vs-GNN
    comparison would be confounded by the input, not the architecture.
    """

    def __init__(self, manifest, cfg=None, h5_path=None, tag_vocab=None,
                 split: str | None = None, n_frames: int = 256, key_suffix: str = ""):
        self.manifest = _read_manifest(manifest)
        if split is not None and "split" in self.manifest.columns:
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.cfg = cfg
        self.h5_path = str(resolve_path(h5_path)) if h5_path else None
        self.key_suffix = key_suffix
        self.n_frames = int(n_frames)
        self.n_mels = int(cfg["audio"]["n_mels"]) if cfg else 128
        self.tag_vocab = list(tag_vocab) if tag_vocab is not None else []
        self.tag_index = {t: i for i, t in enumerate(self.tag_vocab)}
        self._store = None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_store"] = None
        return state

    def _handle(self):
        if self._store is None and self.h5_path:
            import h5py

            self._store = h5py.File(self.h5_path, "r")
        return self._store

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        track_id = str(row["track_id"])
        store = self._handle()
        key = f"{track_id}{self.key_suffix}"
        if store is None or key not in store:
            raise KeyError(f"no cached mel spectrogram for {key!r}")
        mel = np.asarray(store[key][...], dtype=np.float32)
        if mel.shape[0] != self.n_mels and mel.shape[-1] == self.n_mels:
            mel = mel.T

        # centre-crop / edge-pad to a fixed width so the batch is rectangular
        width = mel.shape[1]
        if width >= self.n_frames:
            start = (width - self.n_frames) // 2
            mel = mel[:, start: start + self.n_frames]
        else:
            pad = self.n_frames - width
            mel = np.pad(mel, ((0, 0), (pad // 2, pad - pad // 2)), mode="edge")

        y = np.full(len(self.tag_vocab), -1.0, dtype=np.float32)
        tags = row["y_tags"] if "y_tags" in row else []
        if str(row.get("dataset", "")) in {"mtat", "musiccaps"} or tags:
            y[:] = 0.0
            for tag in tags:
                idx = self.tag_index.get(tag)
                if idx is not None:
                    y[idx] = 1.0
        return (
            torch.from_numpy(mel).unsqueeze(0).float(),
            torch.from_numpy(y).float(),
            track_id,
        )


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def make_loader(ds, cfg, shuffle: bool = False, seed: int = 42, batch_size=None,
                drop_last: bool = False, num_workers=None):
    """PyG ``DataLoader`` with deterministic worker seeding.

    ``persistent_workers`` and ``pin_memory`` come from config; both are off
    automatically when ``num_workers == 0`` (Windows and notebooks) because
    torch errors rather than warns on that combination.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch.utils.data import DataLoader as TorchDataLoader

    train_cfg = cfg["train"] if cfg is not None and "train" in cfg else {}
    workers = int(train_cfg.get("num_workers", 0) if num_workers is None else num_workers)
    batch = int(batch_size or train_cfg.get("batch_size", 8))
    kwargs = dict(
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=get_generator(seed),
    )
    if workers > 0:
        kwargs["pin_memory"] = bool(train_cfg.get("pin_memory", True))
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))

    loader_cls = PyGDataLoader if isinstance(ds, MusicGraphDataset) else TorchDataLoader
    if not isinstance(ds, (MusicGraphDataset, MelSpecDataset)):
        # a plain list of Data objects still wants PyG collation
        sample = ds[0] if len(ds) else None
        from torch_geometric.data import Data as PyGData

        loader_cls = PyGDataLoader if isinstance(sample, PyGData) else TorchDataLoader
    return loader_cls(ds, **kwargs)


def alternating_loader(loader_a: Iterable, loader_b: Iterable, ratio: tuple[int, int] = (1, 1)):
    """Interleave two loaders, cycling the shorter one until the longer ends.

    Task 3 needs this because no track has both MTAT tags and DEAM
    valence/arousal: an epoch that drained MTAT first and DEAM second would let
    the emotion heads drift for thousands of steps with no gradient, then whiplash.
    Alternating keeps both losses live in every optimiser window.
    """
    a_list, b_list = list(loader_a), list(loader_b)
    if not a_list and not b_list:
        return
    if not a_list:
        yield from ((batch, "b") for batch in b_list)
        return
    if not b_list:
        yield from ((batch, "a") for batch in a_list)
        return

    n_a, n_b = ratio
    longer_is_a = len(a_list) >= len(b_list)
    primary, secondary = (a_list, b_list) if longer_is_a else (b_list, a_list)
    primary_tag, secondary_tag = ("a", "b") if longer_is_a else ("b", "a")
    secondary_cycle = cycle(secondary)
    per_primary = max(1, (n_b if longer_is_a else n_a))

    for i, batch in enumerate(primary):
        yield batch, primary_tag
        if (i + 1) % max(1, (n_a if longer_is_a else n_b)) == 0:
            for _ in range(per_primary):
                yield next(secondary_cycle), secondary_tag


def collate_texts(batch, tokenizer, max_length: int = 128, device=None):
    """Tokenise the ``text`` field of a PyG batch into ``input_ids``/``mask``.

    Done in the training loop rather than the Dataset so tokenizer state is
    never pickled into dataloader workers (a common source of a silent
    ``fork``-time hang on the HuggingFace fast tokenizers).
    """
    texts = batch.text if isinstance(batch.text, list) else [batch.text]
    encoded = tokenizer(
        [t if isinstance(t, str) and t else "music" for t in texts],
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
    )
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded["input_ids"], encoded["attention_mask"]
