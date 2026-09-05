#!/usr/bin/env python
"""Check every dataset path before a single feature is extracted.

Run this FIRST. It never crashes on missing data -- it reports it. The most
important thing it prints is the MusicCaps nominal-vs-actual table: roughly half
of MusicCaps' YouTube sources are gone, and the number of survivors in the
AudioSet-eval split *is* the Task 4 retrieval gallery size. R@10 out of 1,400 and
R@10 out of 2,858 are different claims, so the number belongs in the report, not
in a footnote.

    python scripts/verify_datasets.py [--config config.yaml] [--full-decode]
                                      [--sample N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import get_logger, load_config, resolve_path, save_json  # noqa: E402

LOGGER = get_logger("gbmc.verify")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".au"}
MIDI_EXTENSIONS = {".mid", ".midi"}


def _count_files(root: Path, extensions: set[str], limit: int | None = None) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.suffix.lower() in extensions:
            total += 1
            if limit and total >= limit:
                break
    return total


def _sample_decodable(root: Path, extensions: set[str], n: int, seed: int = 42) -> dict:
    """Try to decode ``n`` randomly sampled files and report what happened."""
    if not root.exists():
        return {"sampled": 0, "decoded": 0, "failed": 0, "mean_duration_s": None}
    files = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    if not files:
        return {"sampled": 0, "decoded": 0, "failed": 0, "mean_duration_s": None}

    rng = np.random.default_rng(seed)
    picks = [files[int(i)] for i in rng.permutation(len(files))[: int(n)]]
    durations, failures = [], []
    for path in picks:
        try:
            import librosa

            durations.append(float(librosa.get_duration(path=str(path))))
        except Exception as exc:
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "sampled": len(picks),
        "decoded": len(durations),
        "failed": len(failures),
        "mean_duration_s": float(np.mean(durations)) if durations else None,
        "min_duration_s": float(np.min(durations)) if durations else None,
        "max_duration_s": float(np.max(durations)) if durations else None,
        "failures": failures[:5],
    }


def check_mtat(cfg, sample: int) -> dict:
    paths = cfg["datasets"]["mtat"]
    audio = resolve_path(paths["audio"])
    ann = resolve_path(paths["annotations"])
    info = resolve_path(paths["clip_info"])
    out = {
        "dataset": "MagnaTagATune",
        "root": str(resolve_path(paths["root"])),
        "audio_dir_exists": audio.exists(),
        "annotations_exists": ann.exists(),
        "clip_info_exists": info.exists(),
        "nominal_clips": 25863,
        "nominal_tags": 188,
    }
    out["actual_clips"] = _count_files(audio, AUDIO_EXTENSIONS)
    if ann.exists():
        frame = pd.read_csv(ann, sep="\t")
        out["annotation_rows"] = int(len(frame))
        out["annotation_tag_columns"] = int(len(frame.columns) - 2)
        try:
            from src.splits import reduce_to_top_k_tags

            _, tags = reduce_to_top_k_tags(
                frame, k=int(cfg["tags"]["top_k"]),
                merge_synonyms=bool(cfg["tags"]["merge_synonyms"]),
            )
            out["top_k_tags"] = tags[:10]
        except Exception as exc:
            out["tag_reduction_error"] = str(exc)
    out["decode_check"] = _sample_decodable(audio, AUDIO_EXTENSIONS, sample)
    return out


def check_fma(cfg, sample: int) -> dict:
    paths = cfg["datasets"]["fma"]
    audio = resolve_path(paths["audio"])
    meta = resolve_path(paths["metadata"])
    out = {
        "dataset": "FMA-small",
        "root": str(resolve_path(paths["root"])),
        "audio_dir_exists": audio.exists(),
        "metadata_dir_exists": meta.exists(),
        "nominal_clips": 8000,
        "nominal_genres": 8,
    }
    out["actual_clips"] = _count_files(audio, AUDIO_EXTENSIONS)
    tracks = meta / "tracks.csv"
    if tracks.exists():
        try:
            frame = pd.read_csv(tracks, index_col=0, header=[0, 1])
            small = frame[frame[("set", "subset")] == "small"]
            out["metadata_rows_small"] = int(len(small))
            out["genres"] = sorted(small[("track", "genre_top")].dropna().unique().tolist())
            out["split_counts"] = small[("set", "split")].value_counts().to_dict()
        except Exception as exc:
            out["metadata_error"] = str(exc)
    out["decode_check"] = _sample_decodable(audio, AUDIO_EXTENSIONS, sample)
    return out


def check_deam(cfg, sample: int) -> dict:
    paths = cfg["datasets"]["deam"]
    audio = resolve_path(paths["audio"])
    ann = resolve_path(paths["annotations"])
    out = {
        "dataset": "DEAM",
        "root": str(resolve_path(paths["root"])),
        "audio_dir_exists": audio.exists(),
        "annotations_dir_exists": ann.exists(),
        "nominal_clips": 1802,
    }
    out["actual_clips"] = _count_files(audio, AUDIO_EXTENSIONS)
    static = ann / "annotations averaged per song" / "song_level"
    if static.exists():
        rows = 0
        for path in static.glob("*.csv"):
            frame = pd.read_csv(path)
            frame.columns = [c.strip() for c in frame.columns]
            rows += len(frame)
        out["static_annotation_rows"] = int(rows)
    dynamic = ann / "annotations averaged per song" / "dynamic (per second annotations)"
    out["dynamic_annotations_exist"] = dynamic.exists()
    out["decode_check"] = _sample_decodable(audio, AUDIO_EXTENSIONS, sample)
    return out


def check_musiccaps(cfg, full_decode: bool) -> dict:
    """The one that matters: nominal vs actual, per AudioSet-eval split."""
    from src.splits import build_musiccaps_manifest, musiccaps_survival_summary

    paths = cfg["datasets"]["musiccaps"]
    csv_path = resolve_path(paths["csv"])
    audio = resolve_path(paths["audio"])
    out = {
        "dataset": "MusicCaps",
        "root": str(resolve_path(paths["root"])),
        "csv_exists": csv_path.exists(),
        "audio_dir_exists": audio.exists(),
        "nominal_rows": 5521,
        "nominal_eval": 2858,
        "nominal_train": 2663,
    }
    out["actual_audio_files"] = _count_files(audio, AUDIO_EXTENSIONS)
    if not csv_path.exists():
        return out

    source = pd.read_csv(csv_path)
    out["csv_rows"] = int(len(source))
    out["csv_eval_rows"] = int(source["is_audioset_eval"].sum())
    out["csv_train_rows"] = int((~source["is_audioset_eval"].astype(bool)).sum())

    manifest, log = build_musiccaps_manifest(cfg, verify_decode=full_decode)
    out["usable_rows"] = int(len(manifest))
    out["survival"] = musiccaps_survival_summary(log)
    out["decode_verified"] = bool(full_decode)

    splits_dir = resolve_path(cfg["paths"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    if len(log):
        log.to_csv(splits_dir / "musiccaps_download_log.csv", index=False)
        out["download_log"] = str(splits_dir / "musiccaps_download_log.csv")
    if len(manifest):
        # Rebuild through build_musiccaps_splits, not write_manifest, so the
        # text-variant sidecar is regenerated in the same breath. After a
        # --retry-failed pass the manifest gains rows, and a stale sidecar would
        # silently leave those rows on the RAW caption -- i.e. label leakage.
        from src.splits import build_musiccaps_splits

        refreshed = build_musiccaps_splits(cfg, verify_decode=full_decode)
        out["manifest"] = str(splits_dir / "musiccaps_manifest.csv")
        out["text_variants"] = str(splits_dir / "musiccaps_text_variants.csv")
        out["usable_rows"] = int(len(refreshed))
    return out


def check_lmd(cfg) -> dict:
    root = resolve_path(cfg["datasets"]["lmd"]["root"])
    out = {
        "dataset": "Lakh MIDI Clean",
        "root": str(root),
        "dir_exists": root.exists(),
        "nominal_files": 17256,
    }
    out["actual_files"] = _count_files(root, MIDI_EXTENSIONS)
    out["artist_dirs"] = len([p for p in root.iterdir() if p.is_dir()]) if root.exists() else 0
    return out


def _print_table(report: dict) -> None:
    rows = []
    for key, entry in report["datasets"].items():
        nominal = entry.get("nominal_clips") or entry.get("nominal_rows") or \
            entry.get("nominal_files") or 0
        actual = entry.get("actual_clips") or entry.get("usable_rows") or \
            entry.get("actual_files") or entry.get("actual_audio_files") or 0
        survival = f"{100.0 * actual / nominal:5.1f}%" if nominal else "    -"
        decode = entry.get("decode_check", {})
        rows.append((
            entry.get("dataset", key),
            f"{nominal:>7,}", f"{actual:>7,}", survival,
            f"{decode.get('decoded', '-')}/{decode.get('sampled', '-')}",
            "OK" if actual > 0 else "MISSING",
        ))

    header = ("dataset", "nominal", "actual", "survival", "decoded", "status")
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "  ".join("-" * w for w in widths)
    print("\n" + "=" * len(line))
    print("DATASET VERIFICATION")
    print("=" * len(line))
    print("  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    print(line)
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print(line)

    survival = report["datasets"].get("musiccaps", {}).get("survival", {})
    if survival:
        print("\nMusicCaps survival by split (the Task 4 gallery size):")
        for split in ("train", "eval"):
            nominal = survival.get(f"{split}_nominal", 0)
            survivors = survival.get(f"{split}_survivors", 0)
            rate = 100.0 * survival.get(f"{split}_survival_rate", 0.0)
            print(f"  {split:<6} {survivors:>6,} / {nominal:<6,}  ({rate:5.1f}%)")
        print(f"  -> retrieval gallery size = {survival.get('retrieval_gallery_size', 0):,}")
        counts = survival.get("status_counts", {})
        if counts:
            print("  status counts: " + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))

    missing = [k for k, v in report["datasets"].items()
               if not (v.get("actual_clips") or v.get("usable_rows") or
                       v.get("actual_files") or v.get("actual_audio_files"))]
    if missing:
        print(f"\nMISSING OR EMPTY: {', '.join(missing)}")
        print("Write/run the scripts/download_*.sh scripts to fetch them; nothing "
              "is downloaded automatically.")
    else:
        print("\nAll five corpora are present.")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify every dataset path.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=20,
                        help="how many files per corpus to decode as a spot check")
    parser.add_argument("--full-decode", action="store_true",
                        help="decode-check EVERY MusicCaps clip (slow but exact)")
    parser.add_argument("--json", default=None, help="also write the report here")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    report = {"config": str(cfg.get("_config_path", args.config)), "datasets": {}}

    checks = {
        "mtat": lambda: check_mtat(cfg, args.sample),
        "fma": lambda: check_fma(cfg, args.sample),
        "deam": lambda: check_deam(cfg, args.sample),
        "musiccaps": lambda: check_musiccaps(cfg, args.full_decode),
        "lmd": lambda: check_lmd(cfg),
    }
    for name, check in checks.items():
        try:
            report["datasets"][name] = check()
        except Exception as exc:          # a broken corpus must not stop the report
            LOGGER.warning("check for %s failed: %s", name, exc)
            report["datasets"][name] = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}

    _print_table(report)
    out = args.json or (resolve_path(cfg["paths"]["splits"]) / "dataset_verification.json")
    save_json(report, out)
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
