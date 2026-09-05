#!/usr/bin/env python
"""Rebuild result JSONs from a training log when the output files were lost.

    python scripts/recover_results_from_log.py run.log [--out results/]

Kaggle discards `/kaggle/working` when an interactive session ends, and a
**Quick Save** stores the notebook without re-running it, so its version carries
no output files. The console log is then the only surviving record — but it is a
complete one: every run prints its full result payload, and every epoch prints
its validation metric.

This parses that back into the same JSON files a normal run would have written,
reconstructing `history` from the per-epoch lines. Two fields cannot be
recovered because they are never printed:

* ``thresholds`` — the 50 per-tag decision thresholds. Downstream code falls
  back to 0.5, so per-tag tables will differ slightly from the original run.
* the per-term loss breakdown inside each history entry.

Every recovered file is stamped ``recovered_from_log: true`` so it can never be
mistaken for a first-hand artifact.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, save_json  # noqa: E402

LOGGER = get_logger("gbmc.recover")

RUN_HEADER = re.compile(r"===\s*(?P<tag>[A-Za-z0-9_.\-]+)\s*===")
EPOCH_LINE = re.compile(
    r"task\s+(?P<task>\d+)\s+epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>[-\d.naN]+)\s*\|\s*"
    r"val\s+(?P<metric>[A-Za-z0-9_@]+)=(?P<value>[-\d.naN]+)\s*\|\s*"
    r"(?P<seconds>[\d.]+)s"
)
FREEZE_LINE = re.compile(r"BERT freeze_mode=(?P<mode>\w+) -> (?P<params>\d+) trainable")


def _json_blocks(text: str):
    """Yield every top-level ``{...}`` block that parses as JSON."""
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = text[start: i + 1]
                    try:
                        yield start, json.loads(chunk)   # json.loads accepts NaN
                    except json.JSONDecodeError:
                        pass
                    start = None


def parse(text: str) -> list[dict]:
    """Reconstruct one payload per run found in the log."""
    headers = [(m.start(), m.group("tag")) for m in RUN_HEADER.finditer(text)]
    epochs = [(m.start(), m.groupdict()) for m in EPOCH_LINE.finditer(text)]
    freezes = [(m.start(), m.groupdict()) for m in FREEZE_LINE.finditer(text)]
    blocks = [(pos, obj) for pos, obj in _json_blocks(text)
              if isinstance(obj, dict) and obj.get("task") is not None]

    if not blocks:
        raise SystemExit("no result payloads found in the log")

    recovered = []
    for index, (pos, payload) in enumerate(blocks):
        lower = headers[-1][0] if headers else 0
        tag = None
        for hpos, htag in headers:
            if hpos < pos:
                tag, lower = htag, hpos
            else:
                break
        upper = pos

        history = []
        for epos, fields in epochs:
            if lower < epos < upper:
                history.append({
                    "epoch": int(fields["epoch"]),
                    "seconds": float(fields["seconds"]),
                    "train_loss_total": float(fields["loss"]),
                    f"val_{fields['metric']}": float(fields["value"]),
                })

        modes = [f["mode"] for fpos, f in freezes if lower < fpos < upper]

        payload = dict(payload)
        payload["history"] = history
        payload["thresholds"] = None
        payload["recovered_from_log"] = True
        payload["recovery_note"] = (
            "Rebuilt from the console log; per-tag thresholds were not printed "
            "and are absent. Test metrics are verbatim from the run."
        )
        payload["sweep_tag"] = tag or f"run{index}"
        if modes:
            payload["observed_freeze_modes"] = modes
            payload["effective_freeze_mode"] = modes[-1]

        # a transcription slip would show up here rather than silently
        if history and payload.get("epochs_run") not in (None, len(history)):
            LOGGER.warning(
                "%s: log shows %d epochs but the payload says epochs_run=%s",
                payload["sweep_tag"], len(history), payload.get("epochs_run"),
            )
        recovered.append(payload)
    return recovered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recover results from a log.")
    parser.add_argument("log")
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    runs = parse(text)
    out_dir = ensure_dir(args.out)

    print(f"{'sweep tag':<44}{'macro-F1':>10}{'micro-F1':>10}{'AUC-PR':>9}"
          f"{'epochs':>8}{'freeze':>14}")
    print("-" * 95)
    for payload in runs:
        seed = args.seed if args.seed is not None else payload.get("seed", 42)
        name = f"task{payload['task']}_seed{seed}_{payload['sweep_tag']}.json"
        save_json(payload, out_dir / name)
        test = payload.get("test", {})
        print(f"{payload['sweep_tag']:<44}"
              f"{test.get('macro_f1', float('nan')):>10.4f}"
              f"{test.get('micro_f1', float('nan')):>10.4f}"
              f"{test.get('mean_auc_pr', float('nan')):>9.4f}"
              f"{payload.get('epochs_run', 0):>8}"
              f"{payload.get('effective_freeze_mode', '?'):>14}")

    print(f"\nrecovered {len(runs)} run(s) -> {out_dir}")
    print("Each file is stamped recovered_from_log:true and has no thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
