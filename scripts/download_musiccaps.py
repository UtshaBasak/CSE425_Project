#!/usr/bin/env python
"""Fetch MusicCaps clips with yt-dlp. Resumable, logged, and honest about loss.

MusicCaps ships captions, not audio: each row points at a 10-second window of a
YouTube video, and roughly half of those videos are now deleted, private or
geo-blocked. This script therefore treats failure as the normal case:

* every original ytid gets a row in ``musiccaps_download_log.csv`` with a status
  in ``{ok, missing, corrupt, wrong_duration}``
* the source CSV is never modified -- the usable subset is a *derived* manifest
* already-downloaded clips are skipped, so the script can be killed and resumed
* ``--retry-failed`` re-attempts only the previously failed ids, for the
  recovery pass you will want to run from a different network

    python scripts/download_musiccaps.py [--limit N] [--workers 4]
                                         [--retry-failed] [--sleep 1.0]

NOTE: this is not run during setup. Downloading thousands of YouTube clips takes
hours and is subject to rate limiting; run it deliberately.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils import ensure_dir, get_logger, load_config, resolve_path  # noqa: E402

LOGGER = get_logger("gbmc.musiccaps")

STATUSES = ("ok", "missing", "corrupt", "wrong_duration")


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


def _probe_duration(path: Path) -> float | None:
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as handle:
            return len(handle) / float(handle.samplerate) if handle.samplerate else None
    except Exception:
        pass
    try:
        import librosa

        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def download_clip(ytid: str, start_s: int, end_s: int, out_dir: Path,
                  sleep: float = 0.0, fmt: str = "wav") -> dict:
    """Download and trim one clip. Returns a log row; never raises."""
    stem = f"{ytid}_{start_s}_{end_s}"
    target = out_dir / f"{stem}.{fmt}"
    if target.exists() and target.stat().st_size > 0:
        duration = _probe_duration(target)
        nominal = end_s - start_s
        if duration is None:
            return {"ytid": ytid, "status": "corrupt", "path": str(target),
                    "duration_s": None}
        status = "ok" if abs(duration - nominal) <= 1.0 else "wrong_duration"
        return {"ytid": ytid, "status": status, "path": str(target),
                "duration_s": duration}

    url = f"https://www.youtube.com/watch?v={ytid}"
    command = [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "bestaudio", "-x", "--audio-format", fmt,
        # download only the needed window; --force-keyframes-at-cuts keeps the
        # trim accurate, which matters when the target is exactly 10 seconds
        "--download-sections", f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "-o", str(out_dir / f"{stem}.%(ext)s"),
        url,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {"ytid": ytid, "status": "missing", "path": "", "duration_s": None,
                "error": "timeout"}
    finally:
        if sleep:
            time.sleep(float(sleep))

    if result.returncode != 0 or not target.exists():
        return {"ytid": ytid, "status": "missing", "path": "", "duration_s": None,
                "error": (result.stderr or "").strip()[:200]}
    if target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        return {"ytid": ytid, "status": "corrupt", "path": "", "duration_s": None}

    duration = _probe_duration(target)
    nominal = end_s - start_s
    if duration is None:
        return {"ytid": ytid, "status": "corrupt", "path": str(target), "duration_s": None}
    status = "ok" if abs(duration - nominal) <= 1.0 else "wrong_duration"
    return {"ytid": ytid, "status": status, "path": str(target), "duration_s": duration}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download the MusicCaps audio.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4,
                        help="keep this low; YouTube rate-limits aggressively")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="seconds to pause after each clip")
    parser.add_argument("--format", default="wav", choices=["wav", "mp3", "m4a"])
    parser.add_argument("--retry-failed", action="store_true",
                        help="only re-attempt ids that previously failed")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be downloaded and exit")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    paths = cfg["datasets"]["musiccaps"]
    csv_path = resolve_path(paths["csv"])
    audio_dir = ensure_dir(paths["audio"])
    splits_dir = ensure_dir(cfg["paths"]["splits"])
    log_path = splits_dir / "musiccaps_download_log.csv"

    if not csv_path.exists():
        LOGGER.error("musiccaps-public.csv not found at %s -- download it from "
                     "https://www.kaggle.com/datasets/googleai/musiccaps first", csv_path)
        return 1
    if not _have("yt-dlp"):
        LOGGER.error("yt-dlp is not on PATH (`pip install yt-dlp`)")
        return 1
    if not _have("ffmpeg"):
        LOGGER.error("ffmpeg is not on PATH; yt-dlp cannot trim or transcode without it")
        return 1

    source = pd.read_csv(csv_path)          # read-only: never modified
    todo = source

    if args.retry_failed:
        if not log_path.exists():
            LOGGER.error("--retry-failed needs an existing %s", log_path)
            return 1
        previous = pd.read_csv(log_path)
        failed = set(previous.loc[previous["status"] != "ok", "ytid"].astype(str))
        todo = source[source["ytid"].astype(str).isin(failed)]
        LOGGER.info("retrying %d previously failed ids", len(todo))
    if args.limit:
        todo = todo.head(int(args.limit))

    if args.dry_run:
        print(json.dumps({"would_attempt": int(len(todo)),
                          "audio_dir": str(audio_dir),
                          "log": str(log_path)}, indent=2))
        return 0

    rows: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        pending = {
            pool.submit(download_clip, str(r.ytid), int(r.start_s), int(r.end_s),
                        audio_dir, args.sleep, args.format): str(r.ytid)
            for r in todo.itertuples(index=False)
        }
        for done, future in enumerate(futures.as_completed(pending), start=1):
            rows.append(future.result())
            if done % 50 == 0:
                ok = sum(r["status"] == "ok" for r in rows)
                LOGGER.info("%d/%d attempted, %d ok (%.1f%%)",
                            done, len(pending), ok, 100.0 * ok / done)
                pd.DataFrame(rows).to_csv(log_path.with_suffix(".partial.csv"), index=False)

    # merge with any previous log so the file always covers every original ytid
    fresh = pd.DataFrame(rows)
    full = source[["ytid", "start_s", "end_s", "is_audioset_eval"]].copy()
    full["ytid"] = full["ytid"].astype(str)
    if log_path.exists():
        previous = pd.read_csv(log_path)
        previous["ytid"] = previous["ytid"].astype(str)
        merged = previous.set_index("ytid")
        merged.update(fresh.set_index("ytid"))
        fresh = merged.reset_index()
    log = full.merge(fresh, on="ytid", how="left")
    log["status"] = log["status"].fillna("missing")
    log.to_csv(log_path, index=False)
    log_path.with_suffix(".partial.csv").unlink(missing_ok=True)

    survival = (
        log.assign(ok=log["status"] == "ok")
        .groupby("is_audioset_eval")["ok"].agg(["sum", "count"])
    )
    summary = {"log": str(log_path), "status_counts": log["status"].value_counts().to_dict()}
    for is_eval, row in survival.iterrows():
        key = "eval" if is_eval else "train"
        summary[f"{key}_survivors"] = int(row["sum"])
        summary[f"{key}_nominal"] = int(row["count"])
        summary[f"{key}_survival_rate"] = float(row["sum"] / max(int(row["count"]), 1))
    summary["retrieval_gallery_size"] = summary.get("eval_survivors", 0)
    print(json.dumps(summary, indent=2))
    LOGGER.info("Run `python scripts/verify_datasets.py --full-decode` next to "
                "rebuild the manifest from what actually decodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
