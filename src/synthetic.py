"""Synthetic, contract-compliant data so every module can be built on day one.

This is not a toy: it reproduces the *structural* properties that break naive
implementations, so a module that passes here is very likely to survive real
data.

* **Realistic sentinel patterns.** No track carries both tags and
  valence/arousal, exactly as in the real corpora -- MTAT-like rows get tags and
  ``nan`` emotion, DEAM-like rows get emotion and ``-1`` tags, FMA-like rows get
  only a genre. Any loss that forgets to mask will visibly diverge here.
* **Artist-disjoint splits.** Tracks are grouped under synthetic artists and
  whole artists are assigned to a split, so ``assert_no_leakage`` is exercised.
* **Learnable signal.** Node features are genre prototypes plus temporal drift
  plus noise, and tags are sampled from genre-conditioned probabilities, so a
  working model beats the random baseline and a broken one does not.
* **Long-tailed tag frequencies**, because macro-F1 on a uniform vocabulary
  tells you nothing about the metric's behaviour on the real long tail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audio_features import NODE_FEAT_DIM
from .graph_builder import build_segment_graph
from .splits import MANIFEST_COLUMNS, assert_no_leakage, write_manifest
from .utils import SYNTHETIC, ensure_dir, get_logger, load_config, save_json, set_seed

LOGGER = get_logger("gbmc.synthetic")

__all__ = ["make_synthetic_dataset", "SYNTHETIC_GENRES", "synthetic_tag_vocab"]

SYNTHETIC_GENRES = [
    "electronic", "rock", "classical", "hiphop",
    "jazz", "folk", "ambient", "metal",
]

_TAG_STEMS = [
    "guitar", "drum", "synth", "piano", "vocal", "strings", "beat", "slow",
    "fast", "quiet", "loud", "female vocal", "male vocal", "electronic",
    "acoustic", "distorted", "clean", "brass", "flute", "violin", "choir",
    "ambient", "percussion", "bass", "organ", "harpsichord", "sitar", "banjo",
    "orchestra", "solo", "duet", "instrumental", "melodic", "rhythmic",
    "dark", "bright", "warm", "cold", "dense", "sparse", "repetitive",
    "evolving", "lo-fi", "hi-fi", "reverb", "dry", "swing", "straight",
    "minor key", "major key",
]

_MOOD_WORDS = {
    (True, True): ["euphoric", "triumphant", "energetic"],
    (True, False): ["serene", "warm", "content"],
    (False, True): ["frantic", "aggressive", "tense"],
    (False, False): ["melancholy", "sombre", "wistful"],
}


def synthetic_tag_vocab(n_tags: int = 50) -> list[str]:
    """A stable tag vocabulary of the requested size."""
    if n_tags <= len(_TAG_STEMS):
        return _TAG_STEMS[:n_tags]
    extra = [f"texture {i}" for i in range(n_tags - len(_TAG_STEMS))]
    return _TAG_STEMS + extra


def _genre_prototypes(n_genres: int, rng: np.random.Generator) -> np.ndarray:
    """Well-separated per-genre centroids in the 96-dim feature space."""
    protos = rng.normal(0.0, 1.0, size=(n_genres, NODE_FEAT_DIM))
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    return protos * 3.0


def _tag_probabilities(n_genres: int, n_tags: int, rng: np.random.Generator) -> np.ndarray:
    """Genre-conditioned tag probabilities with a Zipf-like marginal.

    A uniform tag prior would make macro- and micro-F1 nearly identical and hide
    the exact failure mode (rare tags collapsing to all-negative) the metric
    exists to catch.
    """
    base = 0.45 / np.power(np.arange(1, n_tags + 1), 0.75)
    probs = np.tile(base, (n_genres, 1))
    for g in range(n_genres):
        # each genre boosts a distinctive handful of tags
        boosted = rng.choice(n_tags, size=max(3, n_tags // 8), replace=False)
        probs[g, boosted] = np.clip(probs[g, boosted] * 6.0 + 0.25, 0.0, 0.9)
    return np.clip(probs, 0.005, 0.9)


def _caption(genre: str, tags: list[str], valence: float, arousal: float,
             rng: np.random.Generator) -> str:
    mood = _MOOD_WORDS[(valence >= 5.0, arousal >= 5.0)]
    lead = rng.choice(mood)
    listed = ", ".join(tags[:4]) if tags else "sparse instrumentation"
    tempo = "up-tempo" if arousal >= 5.5 else "slow" if arousal < 4.0 else "mid-tempo"
    return (
        f"A {lead} {tempo} {genre} recording featuring {listed}. "
        f"The mix sounds {'bright' if valence >= 5 else 'muted'} and "
        f"{'busy' if arousal >= 5 else 'restrained'}."
    )


def _make_track_features(proto: np.ndarray, n_nodes: int, tag_vector: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """Segment features: genre centroid + a slow drift + a tag imprint + noise.

    The drift is what makes temporal edges informative (adjacent segments really
    are more similar), and the tag imprint is what makes the task learnable at
    all -- without it a GNN and a random baseline would score the same and the
    smoke test would pass on a broken model.
    """
    drift_direction = rng.normal(0, 1, size=NODE_FEAT_DIM)
    drift_direction /= np.linalg.norm(drift_direction) + 1e-8
    positions = np.linspace(-1.0, 1.0, n_nodes)[:, None]

    tag_imprint = np.zeros(NODE_FEAT_DIM, dtype=np.float64)
    active = np.flatnonzero(tag_vector > 0)
    for tag_idx in active:
        # deterministic per-tag direction, so the same tag means the same thing
        # in every track -- this is the signal a classifier can actually learn
        local = np.random.default_rng(10_000 + int(tag_idx)).normal(0, 1, NODE_FEAT_DIM)
        tag_imprint += local / (np.linalg.norm(local) + 1e-8)
    if active.size:
        tag_imprint *= 1.5 / np.sqrt(active.size)

    feats = (
        proto[None, :]
        + positions * drift_direction[None, :] * 1.2
        + tag_imprint[None, :]
        + rng.normal(0.0, 0.45, size=(n_nodes, NODE_FEAT_DIM))
    )
    # a couple of repeated sections, so k-NN similarity edges are not noise
    if n_nodes >= 6:
        chorus = rng.integers(0, n_nodes - 2)
        for repeat in range(1, 3):
            target = (chorus + repeat * (n_nodes // 3)) % n_nodes
            feats[target] = feats[chorus] + rng.normal(0, 0.15, NODE_FEAT_DIM)
    return feats.astype(np.float32)


def _synthetic_mel(feats: np.ndarray, n_mels: int, n_frames: int,
                   rng: np.random.Generator) -> np.ndarray:
    """A mel-shaped view of the same track, for the CNN baseline.

    B2 must see the same information as the GNN, or the comparison measures the
    input rather than the architecture. This upsamples the per-segment features
    onto a mel grid instead of inventing an unrelated signal.
    """
    n_nodes = feats.shape[0]
    projection = np.random.default_rng(7).normal(0, 1, size=(NODE_FEAT_DIM, n_mels))
    projection /= np.linalg.norm(projection, axis=0, keepdims=True) + 1e-8
    per_node = feats @ projection                       # [n_nodes, n_mels]
    idx = np.clip((np.arange(n_frames) * n_nodes) // max(n_frames, 1), 0, n_nodes - 1)
    mel = per_node[idx].T                               # [n_mels, n_frames]
    mel += rng.normal(0, 0.2, size=mel.shape)
    return mel.astype(np.float32)


def make_synthetic_dataset(
    n_tracks: int = 200,
    n_tags: int = 50,
    out_dir: str = "data/processed/synthetic",
    seed: int = 42,
    cfg=None,
    write_mels: bool = True,
) -> dict:
    """Generate graphs, a manifest, a tag vocabulary and train-only norm stats.

    Returns a summary dict; writes:

    ``graphs/<track_id>.pt``    contract-compliant Data objects
    ``manifest.csv``            the frozen manifest schema
    ``tags.json``               the tag vocabulary
    ``norm_stats.json``         mean/std computed on the **train split only**
    ``mels.h5``                 mel cache for the CNN baseline
    ``summary.json``            label coverage, so the sentinel mix is visible
    """
    cfg = cfg if cfg is not None else load_config("config.yaml")
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out = ensure_dir(out_dir)
    graph_dir = ensure_dir(out / "graphs")
    tag_vocab = synthetic_tag_vocab(n_tags)
    genres = SYNTHETIC_GENRES
    protos = _genre_prototypes(len(genres), rng)
    tag_probs = _tag_probabilities(len(genres), n_tags, rng)

    seg_cfg = cfg["segmentation"]
    min_nodes = int(seg_cfg.get("min_nodes", 4))
    max_nodes = int(seg_cfg.get("max_nodes", 32))

    # ---- artists first, so the split is artist-disjoint by construction ---- #
    n_artists = max(8, n_tracks // 3)
    artist_genre = rng.integers(0, len(genres), size=n_artists)
    artist_split = _assign_artist_splits(n_artists, rng)
    # Dataset membership is per artist, so a corpus's sentinel pattern is never
    # split across one artist's discography, and it is stratified *within* each
    # split so every split contains tag-only, VA-only and genre-only tracks.
    artist_dataset = _assign_artist_datasets(artist_split, rng)
    # Round-robin track -> artist so no artist ends up with zero tracks (which
    # would silently empty a split on small runs).
    track_artists = np.array([i % n_artists for i in range(n_tracks)])
    rng.shuffle(track_artists)

    rows, summary_counts = [], {"mtat": 0, "deam": 0, "musiccaps": 0, "fma": 0}
    mels: dict[str, np.ndarray] = {}

    for i in range(n_tracks):
        artist = int(track_artists[i])
        genre_idx = int(artist_genre[artist])
        dataset = str(artist_dataset[artist])
        split = artist_split[artist]
        track_id = f"syn_{i:04d}"
        n_nodes = int(rng.integers(min_nodes, max_nodes + 1))

        tag_vector = (rng.random(n_tags) < tag_probs[genre_idx]).astype(np.float32)
        if tag_vector.sum() == 0:
            tag_vector[int(rng.integers(0, n_tags))] = 1.0
        positive = [tag_vocab[j] for j in np.flatnonzero(tag_vector)]

        # valence/arousal on DEAM's 1-9 scale, correlated with genre
        centre = 5.0 + 2.0 * np.sin(genre_idx * 1.1)
        valence = float(np.clip(rng.normal(centre, 1.0), 1.0, 9.0))
        arousal = float(np.clip(rng.normal(10.0 - centre, 1.0), 1.0, 9.0))
        feats = _make_track_features(protos[genre_idx], n_nodes, tag_vector, rng)

        # ---- the sentinel pattern, per corpus --------------------------- #
        if dataset == "mtat":
            labels = dict(y_tags=tag_vector, y_genre=-1,
                          y_valence=float("nan"), y_arousal=float("nan"))
            text = ", ".join(positive)
        elif dataset == "deam":
            labels = dict(y_tags=np.full(n_tags, -1.0, dtype=np.float32), y_genre=-1,
                          y_valence=valence, y_arousal=arousal)
            text = _caption(genres[genre_idx], positive, valence, arousal, rng)
        elif dataset == "musiccaps":
            labels = dict(y_tags=tag_vector, y_genre=-1,
                          y_valence=float("nan"), y_arousal=float("nan"))
            text = _caption(genres[genre_idx], positive, valence, arousal, rng)
        else:  # fma: genre only
            labels = dict(y_tags=np.full(n_tags, -1.0, dtype=np.float32),
                          y_genre=genre_idx,
                          y_valence=float("nan"), y_arousal=float("nan"))
            text = f"{genres[genre_idx]} track by synthetic artist {artist:03d}."
        summary_counts[dataset] += 1

        data = build_segment_graph(
            feats, cfg,
            track_id=track_id,
            artist_id=f"syn_artist_{artist:03d}",
            dataset=dataset,
            provenance=SYNTHETIC,
            split=split,
            text=text,
            n_tags=n_tags,
            **labels,
        )
        # genre is kept as an unsupervised colouring for the t-SNE panels even
        # where it is not a training target
        data.true_genre = torch.tensor([genre_idx], dtype=torch.long)
        torch.save(data, graph_dir / f"{track_id}.pt")

        if write_mels:
            mels[track_id] = _synthetic_mel(
                feats, int(cfg["audio"]["n_mels"]), 256, rng
            )

        rows.append({
            "track_id": track_id,
            "artist_id": f"syn_artist_{artist:03d}",
            "audio_path": "",
            "text": text,
            "split": split,
            "y_genre": genre_idx if dataset == "fma" else -1,
            "y_tags": json.dumps(positive if dataset in {"mtat", "musiccaps"} else []),
            "y_valence": valence if dataset == "deam" else np.nan,
            "y_arousal": arousal if dataset == "deam" else np.nan,
            "duration_s": round(n_nodes * 1.5, 2),
            "dataset": dataset,
            "provenance": SYNTHETIC,
            "true_genre": genre_idx,
        })

    manifest = pd.DataFrame(rows)
    assert_no_leakage(manifest)

    manifest_path = out / "manifest.csv"
    frame = manifest.copy()
    frame[MANIFEST_COLUMNS + ["dataset", "provenance", "true_genre"]].to_csv(
        manifest_path, index=False
    )

    with open(out / "tags.json", "w", encoding="utf-8") as fh:
        json.dump({"tags": tag_vocab, "genres": genres}, fh, indent=2)

    # ---- norm stats: TRAIN SPLIT ONLY -------------------------------------- #
    train_ids = manifest.loc[manifest["split"] == "train", "track_id"].tolist()
    train_feats = [
        torch.load(graph_dir / f"{t}.pt", weights_only=False).x.numpy() for t in train_ids
    ]
    stacked = np.concatenate(train_feats, axis=0) if train_feats else np.zeros((1, NODE_FEAT_DIM))
    std = stacked.std(axis=0)
    std[std < 1e-6] = 1.0
    save_json(
        {
            "mean": stacked.mean(axis=0).astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "n_segments": int(stacked.shape[0]),
            "split": "train",
            "dim": NODE_FEAT_DIM,
        },
        out / "norm_stats.json",
    )

    if write_mels:
        import h5py

        with h5py.File(out / "mels.h5", "w") as store:
            for key, value in mels.items():
                store.create_dataset(key, data=value.astype(np.float16),
                                     dtype="float16", compression="lzf")
            store.attrs["n_mels"] = int(cfg["audio"]["n_mels"])

    summary = {
        "provenance": SYNTHETIC,
        "n_tracks": int(len(manifest)),
        "n_tags": int(n_tags),
        "n_genres": len(genres),
        "n_artists": int(manifest["artist_id"].nunique()),
        "out_dir": str(out),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "dataset_counts": summary_counts,
        "tracks_with_tags": int((manifest["dataset"].isin(["mtat", "musiccaps"])).sum()),
        "tracks_with_valence_arousal": int((manifest["dataset"] == "deam").sum()),
        "tracks_with_genre": int((manifest["dataset"] == "fma").sum()),
        "tracks_with_both_tags_and_va": 0,   # by construction, as in the real data
        "seed": int(seed),
    }
    save_json(summary, out / "summary.json")
    LOGGER.info("synthetic dataset ready: %s", summary)
    return summary


def _assign_artist_splits(n_artists: int, rng: np.random.Generator) -> list[str]:
    """70/15/15 over artists, guaranteeing every split is non-empty."""
    order = rng.permutation(n_artists)
    n_val = max(1, int(round(0.15 * n_artists)))
    n_test = max(1, int(round(0.15 * n_artists)))
    n_train = max(1, n_artists - n_val - n_test)
    assignment = [""] * n_artists
    for rank, artist in enumerate(order):
        if rank < n_train:
            assignment[artist] = "train"
        elif rank < n_train + n_val:
            assignment[artist] = "val"
        else:
            assignment[artist] = "test"
    return assignment


def _assign_artist_datasets(artist_split: list[str], rng: np.random.Generator) -> list[str]:
    """Stratify the four corpora within each split.

    Every split therefore holds tag-only (mtat/musiccaps), VA-only (deam) and
    genre-only (fma) tracks, which is what makes the masked multi-task loss and
    the alternating loader exercisable on any split.
    """
    order = ["mtat", "musiccaps", "deam", "fma"]
    assignment = [""] * len(artist_split)
    for split in ("train", "val", "test"):
        members = [i for i, s in enumerate(artist_split) if s == split]
        rng.shuffle(members)
        for rank, artist in enumerate(members):
            assignment[artist] = order[rank % len(order)]
    return assignment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-tracks", type=int, default=None)
    parser.add_argument("--n-tags", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    syn = cfg.get("synthetic", {})
    summary = make_synthetic_dataset(
        n_tracks=args.n_tracks or int(syn.get("n_tracks", 200)),
        n_tags=args.n_tags or int(syn.get("n_tags", 50)),
        out_dir=args.out_dir or str(syn.get("out_dir", "data/processed/synthetic")),
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 42)),
        cfg=cfg,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
