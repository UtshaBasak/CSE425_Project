#!/usr/bin/env python
"""Recompute Task 3 metrics on the corpora the run was supposed to evaluate.

    python scripts/recompute_task3_metrics.py [--write]

**The bug.** `run_task3` filtered corpora for the *training* loaders and not for
val/test:

    tag_loader = make_loader(bundle.dataset("train", corpora_for(cfg, "tag")), ...)
    "val":  make_loader(bundle.dataset("val"),  ...)   # no corpora argument
    "test": make_loader(bundle.dataset("test"), ...)   # no corpora argument

So an MTAT run was scored over 7,079 test rows instead of 3,775. FMA and DEAM
rows are harmless -- `datasets.py` gives any corpus outside {mtat, musiccaps}
the -1 sentinel, which the metrics mask out. MusicCaps is not harmless: it *is*
in that set, so its 2,503 rows arrive as confident zeros against the MTAT
vocabulary and count as true negatives on every tag. `corpora_for`'s own
docstring warns about exactly this.

**Why this can be fixed without retraining.** The score matrices are saved, and
`MusicGraphDataset` preserves manifest order while the val/test loaders use
`shuffle=False`. So row *i* of `test_y_true` is row *i* of the split's manifest
slice, and the corpus of every row is recoverable. The script asserts that
mapping against the sentinel pattern before using it and refuses to write
anything if it does not hold exactly.

**What this cannot fix.** Early stopping and checkpoint selection used the
contaminated validation metric, so these are the right numbers for a model that
was selected on the wrong signal. That residue needs a re-run; it is recorded in
the output as `selection_signal_contaminated: true` rather than papered over.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.metrics as M                                          # noqa: E402
from src.train import DataBundle, corpora_for                    # noqa: E402
from src.utils import get_logger, load_config, project_root      # noqa: E402

LOGGER = get_logger("gbmc.recompute")

#: which corpora each Task 3 run_tag was meant to evaluate on
TAG_CORPUS = {"mtat": "mtat", "musiccaps": "musiccaps"}


def _corpora_for_tag(run_tag: str, cfg) -> set[str]:
    corpus = next((c for k, c in TAG_CORPUS.items() if run_tag.startswith(k)), None)
    if corpus is None:
        return set()
    return {corpus} | set(corpora_for(cfg, "emotion"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="update the result JSONs in place (default: report only)")
    args = parser.parse_args(argv)

    root = project_root()
    cfg = load_config("config.yaml")
    bundle = DataBundle(cfg, False)
    man = bundle.manifest
    order = {s: man[man["split"] == s]["dataset"].to_numpy() for s in ("val", "test")}

    scores = sorted((root / "results" / "scores").glob("task3_seed*_scores.npz"))
    if not scores:
        LOGGER.error("no Task 3 score matrices found")
        return 1

    print(f"{'run':<44}{'published':>10}{'corrected':>11}{'delta':>9}  rows")
    print("-" * 84)
    changed, skipped = [], []

    for path in scores:
        name = path.stem.replace("_scores", "")
        run_tag = name.split("_", 2)[2] if name.count("_") >= 2 else ""
        keep = _corpora_for_tag(run_tag, cfg)
        if not keep:
            skipped.append((name, "unrecognised run_tag"))
            continue

        payload = np.load(path, allow_pickle=True)
        needed = {f"{s}_{k}" for s in ("val", "test") for k in ("y_true", "y_score")}
        if not needed.issubset(set(payload.files)):
            skipped.append((name, "score matrix missing val/test arrays"))
            continue

        cut = {}
        ok = True
        for split in ("val", "test"):
            y_true = np.asarray(payload[f"{split}_y_true"], dtype=float)
            y_score = np.asarray(payload[f"{split}_y_score"], dtype=float)
            datasets = order[split]
            if len(datasets) != len(y_true):
                ok = False
                skipped.append((name, f"{split} length {len(y_true)} != manifest "
                                      f"{len(datasets)}"))
                break
            # The order assumption has to be proved, not assumed: rows outside
            # {mtat, musiccaps} must be exactly the all -1 rows.
            masked = (y_true == -1).all(axis=1)
            expect = ~np.isin(datasets, list({"mtat", "musiccaps"}))
            if not np.array_equal(masked, expect):
                ok = False
                skipped.append((name, "row order does not match the sentinel "
                                      "pattern -- refusing to guess"))
                break
            sel = np.isin(datasets, list(keep)) & ~masked
            cut[split] = (y_true[sel], y_score[sel], int(len(y_true)), int(sel.sum()))
        if not ok:
            continue

        v_true, v_score, _, _ = cut["val"]
        t_true, t_score, n_before, n_after = cut["test"]
        thresholds = M.tune_thresholds(v_true, v_score)
        fixed = {
            "macro_f1": float(M.macro_f1(t_true, t_score, thresholds=thresholds)),
            "micro_f1": float(M.micro_f1(t_true, t_score, thresholds=thresholds)),
            "mean_auc_pr": float(M.mean_auc_pr(t_true, t_score)),
            "macro_f1_fixed_half": float(M.macro_f1(t_true, t_score, thresholds=0.5)),
            "n_test_rows": n_after,
            "n_test_rows_before_fix": n_before,
            "evaluated_corpora": sorted(keep),
            "selection_signal_contaminated": True,
        }

        result_path = root / "results" / f"{name}.json"
        published = None
        if result_path.exists():
            published = json.loads(result_path.read_text(encoding="utf-8")) \
                .get("test", {}).get("macro_f1")
        delta = fixed["macro_f1"] - published if published is not None else float("nan")
        print(f"{name:<44}"
              f"{'--' if published is None else f'{published:.4f}':>10}"
              f"{fixed['macro_f1']:>11.4f}{delta:>+9.4f}  {n_before}->{n_after}")
        changed.append((name, published, fixed))

        if args.write and result_path.exists():
            payload_json = json.loads(result_path.read_text(encoding="utf-8"))
            payload_json.setdefault("test_uncorrected", dict(payload_json.get("test", {})))
            payload_json["test"].update(fixed)
            payload_json["evaluation_corrected"] = {
                "reason": "val/test loaders were not filtered by corpus; "
                          "MusicCaps rows scored against the MTAT vocabulary "
                          "(and vice versa) as confident negatives",
                "recomputed_from": f"results/scores/{path.name}",
                "thresholds_retuned_on": "filtered validation split",
            }
            result_path.write_text(json.dumps(payload_json, indent=2), encoding="utf-8")

    print("-" * 84)
    deltas = [f["macro_f1"] - p for _, p, f in changed if p is not None]
    if deltas:
        print(f"{len(changed)} runs corrected | mean delta {np.mean(deltas):+.4f} "
              f"| range {np.min(deltas):+.4f} to {np.max(deltas):+.4f}")
    for name, why in skipped:
        LOGGER.warning("skipped %s: %s", name, why)
    if args.write:
        print("\nresult JSONs updated; the pre-fix values are kept under "
              "`test_uncorrected`")
    else:
        print("\ndry run -- pass --write to update the result JSONs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
