#!/usr/bin/env python
"""Build segment / chord / hetero graphs for every dataset, resumably.

    python scripts/build_graphs.py [--datasets mtat fma deam musiccaps]
                                   [--kinds segment chord hetero]
                                   [--limit N] [--overwrite]

**Item-level resumable**: a track whose `.pt` already exists is skipped, so this
can be killed and restarted at any point and only redoes the file in flight.
Each graph is written atomically (temp file then `os.replace`), because a
half-written `.pt` that `torch.load` chokes on is worse than a missing one --
the missing one is retried automatically.

Chord graphs are derived from the chroma block of the 96-dim segment feature
(indices 72:84), not from the raw audio again: re-decoding 25k mp3s to recover
chroma we already computed would add hours for nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.audio_features import NODE_FEAT_DIM  # noqa: E402
from src.chords import chords_to_segment_map, estimate_chords  # noqa: E402
from src.graph_builder import (  # noqa: E402
    build_chord_graph,
    build_hetero_graph,
    build_segment_graph,
)
from src.utils import (  # noqa: E402
    REAL,
    atomic_torch_save,
    ensure_dir,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_json,
)

LOGGER = get_logger("gbmc.build_graphs")

KINDS = ("segment", "chord", "hetero")

#: chroma mean occupies [72:84] of the canonical 96-dim feature (see
#: audio_features.FEATURE_LAYOUT); slicing it is exact, not an approximation.
CHROMA_SLICE = slice(72, 84)


def _tag_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (TypeError, ValueError):
        return []


def _labels_from_row(row, tag_vocab: list[str]) -> dict:
    """Contract labels for one manifest row, honouring the sentinel rule."""
    dataset = str(row.get("dataset", ""))
    tags = _tag_list(row.get("y_tags"))

    if tag_vocab and (dataset in {"mtat", "musiccaps"} or tags):
        index = {t: i for i, t in enumerate(tag_vocab)}
        vector = np.zeros(len(tag_vocab), dtype=np.float32)
        for tag in tags:
            if tag in index:
                vector[index[tag]] = 1.0
        y_tags = vector
    elif tag_vocab:
        # corpus carries no tag vocabulary at all -> -1 sentinel, never zeros
        y_tags = np.full(len(tag_vocab), -1.0, dtype=np.float32)
    else:
        y_tags = None

    def _num(key, default):
        try:
            value = float(row.get(key))
            return value if np.isfinite(value) else default
        except (TypeError, ValueError):
            return default

    return {
        "y_tags": y_tags,
        "n_tags": len(tag_vocab) or None,
        "y_genre": int(_num("y_genre", -1)),
        "y_valence": _num("y_valence", float("nan")),
        "y_arousal": _num("y_arousal", float("nan")),
        "track_id": str(row["track_id"]),
        "artist_id": str(row.get("artist_id", "")),
        "dataset": dataset,
        "provenance": str(row.get("provenance", REAL)),
        "split": str(row.get("split", "")),
        "text": str(row.get("text", "") or ""),
    }


def _chord_sequence(feats: np.ndarray, cfg) -> list[str]:
    """Per-segment chord labels from the chroma block of the node features."""
    chroma = feats[:, CHROMA_SLICE].T                      # [12, num_segments]
    chroma = np.clip(chroma - chroma.min(), 0.0, None)     # chroma means can be < 0
    return estimate_chords(chroma, cfg)


def build_graphs_for_manifest(manifest, cfg, out_dir, h5_path=None, graph_dir=None,
                              tag_vocab=None, norm_stats=None,
                              kinds=("segment",), overwrite: bool = False,
                              log_every: int = 500) -> dict:
    """Build graphs for every row, skipping any whose output already exists.

    ``graph_dir`` re-reads pre-built ``.pt`` graphs instead of the HDF5 cache;
    that is how the synthetic dataset and the resumability test feed this
    function without a feature cache.
    """
    import h5py

    manifest = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
    tag_vocab = list(tag_vocab or [])
    kinds = tuple(kinds)
    for kind in kinds:
        if kind not in KINDS:
            raise ValueError(f"unknown graph kind {kind!r}; expected {KINDS}")

    out_dirs = {kind: ensure_dir(Path(out_dir) / kind if len(kinds) > 1 else out_dir)
                for kind in kinds}
    stats = {"total": int(len(manifest)), "written": 0, "skipped": 0, "failed": 0}
    failures: list[dict] = []

    store = None
    if h5_path is not None and Path(resolve_path(h5_path)).exists():
        store = h5py.File(resolve_path(h5_path), "r")

    try:
        for n, row in enumerate(manifest.to_dict("records"), start=1):
            track_id = str(row["track_id"])
            targets = {k: out_dirs[k] / f"{track_id}.pt" for k in kinds}
            if not overwrite and all(p.exists() for p in targets.values()):
                stats["skipped"] += 1
                continue

            try:
                if graph_dir is not None:
                    source = Path(graph_dir) / f"{track_id}.pt"
                    feats = torch.load(source, weights_only=False).x.numpy()
                elif store is not None and track_id in store:
                    feats = np.asarray(store[track_id][...], dtype=np.float32)
                else:
                    raise KeyError(f"no cached features for {track_id!r}")

                if norm_stats is not None:
                    from src.audio_features import apply_norm

                    feats = apply_norm(feats, norm_stats)
                if feats.shape[1] != NODE_FEAT_DIM:
                    raise AssertionError(
                        f"{track_id}: features are {feats.shape[1]}-dim, "
                        f"contract requires {NODE_FEAT_DIM}"
                    )

                labels = _labels_from_row(row, tag_vocab)
                segment = build_segment_graph(feats, cfg, **labels)
                assert segment.x.shape[1] == NODE_FEAT_DIM

                chord = chord_map = None
                if {"chord", "hetero"} & set(kinds):
                    sequence = _chord_sequence(feats, cfg)
                    chord = build_chord_graph(sequence, feats[:, CHROMA_SLICE].T,
                                              cfg, **labels)
                    chord_map = chords_to_segment_map(
                        sequence, int(segment.num_nodes), chord.chord_names)

                for kind in kinds:
                    if kind == "segment":
                        atomic_torch_save(segment, targets[kind])
                    elif kind == "chord":
                        atomic_torch_save(chord, targets[kind])
                    else:
                        atomic_torch_save(
                            build_hetero_graph(segment, chord, chord_map), targets[kind])
                stats["written"] += 1
            except Exception as exc:
                stats["failed"] += 1
                failures.append({"track_id": track_id, "error": f"{type(exc).__name__}: {exc}"})

            if log_every and n % log_every == 0:
                LOGGER.info("  %d/%d (%d written, %d skipped, %d failed)",
                            n, stats["total"], stats["written"], stats["skipped"],
                            stats["failed"])
    finally:
        if store is not None:
            store.close()

    stats["failures"] = failures[:20]
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build graphs for every dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--datasets", nargs="*",
                        default=["mtat", "fma", "deam", "musiccaps"])
    parser.add_argument("--kinds", nargs="*", default=["segment"], choices=list(KINDS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    splits_dir = resolve_path(cfg["paths"]["splits"])
    processed = resolve_path(cfg["paths"]["processed"])
    out_root = Path(args.out_root) if args.out_root else processed / "graphs"

    vocab_path = splits_dir / "tag_vocab.json"
    tag_vocab = []
    if vocab_path.exists():
        payload = json.loads(vocab_path.read_text(encoding="utf-8"))
        tag_vocab = list(payload["tags"] if isinstance(payload, dict) else payload)

    summary = {}
    for name in args.datasets:
        manifest_path = splits_dir / f"{name}_manifest.csv"
        if not manifest_path.exists():
            LOGGER.warning("no manifest for %s; run `make splits` first", name)
            continue
        manifest = pd.read_csv(manifest_path)
        manifest["dataset"] = manifest.get("dataset", name)
        if args.limit:
            manifest = manifest.head(int(args.limit))

        h5_path = processed / f"features_{name}.h5"
        if not h5_path.exists():
            h5_path = processed / "features.h5"
        stats_path = processed / f"norm_stats_{name}.json"
        if not stats_path.exists():
            stats_path = processed / "norm_stats.json"
        norm_stats = None
        if stats_path.exists():
            norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
            if norm_stats.get("split") != "train":
                raise RuntimeError(
                    f"{stats_path} was not computed on the train split; refusing to run"
                )

        LOGGER.info("building %s graphs for %s (%d rows) -> %s",
                    "/".join(args.kinds), name, len(manifest), out_root / name)
        stats = build_graphs_for_manifest(
            manifest, cfg, out_root / name, h5_path=h5_path, tag_vocab=tag_vocab,
            norm_stats=norm_stats, kinds=tuple(args.kinds), overwrite=args.overwrite,
        )
        summary[name] = stats
        LOGGER.info("%s: %s", name, {k: v for k, v in stats.items() if k != "failures"})

    save_json(summary, processed / "graph_build_summary.json")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "failures"}
                      for k, v in summary.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
