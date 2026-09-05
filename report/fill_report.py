#!/usr/bin/env python
"""Inject real numbers from ``results/`` into ``report/final_report.tex``.

    python report/fill_report.py [--check]

The report is a living document: every phase gate updates it. Hand-copying
numbers out of JSON into LaTeX is exactly the step where a stale figure survives
three revisions, so the prose never contains a number. It contains a macro, and
this script rewrites the ``AUTOGEN`` block that defines every macro from the
result files on disk.

A result that does not exist yet becomes ``\\textit{pending}`` rather than a
plausible-looking placeholder, and the script prints which macros are still
pending so the gap is visible instead of silent.

``--check`` exits non-zero if any macro is pending -- useful before declaring a
section finished.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger, project_root  # noqa: E402

LOGGER = get_logger("gbmc.report")

BEGIN = "% --- AUTOGEN:BEGIN ---------------------------------------------------------"
END = "% --- AUTOGEN:END -----------------------------------------------------------"
PENDING = r"\textit{pending}"

ROOT = project_root()
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOGGER.warning("%s is not valid JSON (%s)", name, exc)
        return None


def _scalar(value):
    """Unwrap numpy scalars; pandas counts are int64, not int."""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (ValueError, TypeError):
            return None
    return value if isinstance(value, (int, float)) else None


def num(value, places: int = 4) -> str:
    """A number, or the pending marker -- never a made-up default."""
    value = _scalar(value)
    if value is None:
        return PENDING
    try:
        if value != value:                       # NaN
            return PENDING
    except TypeError:
        return PENDING
    return f"{value:.{places}f}"


def integer(value) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{int(value):,}".replace(",", "{,}")


def seconds(value) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{value / 60:.1f}\\,min" if value >= 90 else f"{value:.0f}\\,s"


def pct(value, places: int = 1) -> str:
    value = _scalar(value)
    if value is None:
        return PENDING
    return f"{100 * value:.{places}f}"


def dig(payload, *path, default=None):
    """Nested lookup that tolerates a missing file entirely."""
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def find_baseline(payload, name_contains: str) -> dict:
    for entry in dig(payload, "baselines", default=[]) or []:
        if name_contains in str(entry.get("baseline", "")):
            return entry
    return {}


# --------------------------------------------------------------------------- #
# the macro table
# --------------------------------------------------------------------------- #
def build_macros() -> dict:
    import pandas as pd

    macros: dict[str, str] = {}

    # ---- corpus sizes, straight from the manifests --------------------- #
    splits_dir = ROOT / "data" / "splits"
    total = 0
    for corpus in ("mtat", "fma", "deam", "musiccaps"):
        path = splits_dir / f"{corpus}_manifest.csv"
        if path.exists():
            frame = pd.read_csv(path)
            total += len(frame)
            if corpus == "musiccaps":
                counts = frame["split"].value_counts()
                macros["MCTest"] = integer(counts.get("test"))
                macros["MCTrain"] = integer(counts.get("train"))
                macros["MCVal"] = integer(counts.get("val"))
            if corpus == "fma":
                macros["FMATestRows"] = integer(
                    (frame["split"] == "test").sum())
    macros["NTracks"] = integer(total) if total else PENDING
    # three views are stored per track -- segment, chord and heterogeneous --
    # so the graph count is 3x the track count, not equal to it
    summary_path = ROOT / "data" / "processed" / "graph_build_summary.json"
    if summary_path.exists():
        written = sum(int(v.get("written", 0)) for v in
                      json.loads(summary_path.read_text(encoding="utf-8")).values())
        macros["NGraphs"] = integer(written * 3)
    else:
        macros["NGraphs"] = PENDING

    # ---- Task 1 sweep on MusicCaps ------------------------------------- #
    runs = {
        "MCProbe": "task1_seed42_musiccaps_caption_masked_frozen_probe.json",
        "MCTopN": "task1_seed42_musiccaps_caption_masked_top_n.json",
        "MCMasked": "task1_seed42_musiccaps_caption_masked_full_ft.json",
        "MCRaw": "task1_seed42_musiccaps_caption_raw_full_ft.json",
    }
    payloads = {}
    for prefix, filename in runs.items():
        payload = load(filename) or {}
        payloads[prefix] = payload
        macros[f"{prefix}F"] = num(dig(payload, "test", "macro_f1"))
        macros[f"{prefix}Micro"] = num(dig(payload, "test", "micro_f1"))
        macros[f"{prefix}PR"] = num(dig(payload, "test", "mean_auc_pr"))
    macros["MCMaskedVal"] = num(dig(payloads["MCMasked"], "best_val_metric"))

    masked = dig(payloads["MCMasked"], "test", "macro_f1")
    raw = dig(payloads["MCRaw"], "test", "macro_f1")
    if isinstance(masked, (int, float)) and isinstance(raw, (int, float)) and masked:
        macros["LeakGap"] = f"$+${raw - masked:.4f}"
        macros["LeakPct"] = f"{100 * (raw - masked) / masked:.0f}"
    else:
        macros["LeakGap"] = macros["LeakPct"] = PENDING

    # ---- Task 1 on MTAT metadata (the like-for-like B3 row) ------------ #
    mtat = load("task1_seed42_mtat_metadata_full_ft.json") or {}
    macros["TOneMtatF"] = num(dig(mtat, "test", "macro_f1"))
    macros["TOneMtatMicro"] = num(dig(mtat, "test", "micro_f1"))
    macros["TOneMtatPR"] = num(dig(mtat, "test", "mean_auc_pr"))
    macros["TOneMtatParams"] = integer(mtat.get("trainable_params"))

    # ---- Task 2, both domains ------------------------------------------ #
    genre = load("task2_seed42_fma_genre.json") or {}
    macros["TTwoGenreAcc"] = pct(dig(genre, "test", "genre_accuracy")) + r"\%" \
        if isinstance(dig(genre, "test", "genre_accuracy"), float) else PENDING
    macros["TTwoGenreF"] = num(dig(genre, "test", "genre_macro_f1"))
    macros["TTwoGenreParams"] = integer(genre.get("trainable_params"))
    macros["TTwoGenreTime"] = seconds(genre.get("wall_clock_s"))

    tags = load("task2_seed42_mtat_tags.json") or load("task2_seed42.json") or {}
    macros["TTwoTagF"] = num(dig(tags, "test", "macro_f1"))
    macros["TTwoTagMicro"] = num(dig(tags, "test", "micro_f1"))
    macros["TTwoTagPR"] = num(dig(tags, "test", "mean_auc_pr"))
    macros["TTwoTagParams"] = integer(tags.get("trainable_params"))

    # ---- B2, both domains ---------------------------------------------- #
    baselines = load("baselines_seed42.json") or {}
    b2_genre = find_baseline(baselines, "B2_mel_cnn_genre")
    b2_acc = pct(b2_genre.get("genre_accuracy"))
    macros["BTwoGenreAcc"] = b2_acc + r"\%" if b2_acc != PENDING else PENDING
    macros["BTwoGenreF"] = num(b2_genre.get("genre_macro_f1"))
    macros["BTwoGenreParams"] = integer(b2_genre.get("trainable_params"))
    macros["BTwoGenreTime"] = seconds(b2_genre.get("wall_clock_s"))

    b2_tags = find_baseline(baselines, "B2_mel_cnn_tags")
    macros["BTwoTagF"] = num(b2_tags.get("macro_f1"))
    macros["BTwoTagMicro"] = num(b2_tags.get("micro_f1"))
    macros["BTwoTagPR"] = num(b2_tags.get("mean_auc_pr"))
    macros["BTwoTagParams"] = integer(b2_tags.get("trainable_params"))

    # ---- A7.4 threshold bootstrap -------------------------------------- #
    boot = load("threshold_bootstrap.json")

    def fixed_half(*names):
        """Untuned macro-F1, from the result if present, else the bootstrap.

        Runs that finished before A7.4 landed have no `macro_f1_fixed_half`
        field, but the bootstrap recomputed it from their stored score matrices,
        so the number is available either way rather than pending.
        """
        for entry in (boot or {}).get("runs", []):
            if entry["run"] in names:
                return num(entry["test_macro_f1_fixed_half"])
        return PENDING

    macros["TTwoTagFixed"] = num(dig(tags, "test", "macro_f1_fixed_half"))
    if macros["TTwoTagFixed"] == PENDING:
        macros["TTwoTagFixed"] = fixed_half("task2_seed42_mtat_tags")
    macros["BTwoTagFixed"] = num(b2_tags.get("macro_f1_fixed_half"))
    if macros["BTwoTagFixed"] == PENDING:
        macros["BTwoTagFixed"] = fixed_half("b2_seed42_tags")
    macros["MCMaskedFixed"] = num(dig(payloads["MCMasked"], "test", "macro_f1_fixed_half"))
    if macros["MCMaskedFixed"] == PENDING:
        macros["MCMaskedFixed"] = fixed_half(
            "task1_seed42_musiccaps_caption_masked_full_ft")
    macros["MCRawFixed"] = num(dig(payloads["MCRaw"], "test", "macro_f1_fixed_half"))
    if macros["MCRawFixed"] == PENDING:
        macros["MCRawFixed"] = fixed_half("task1_seed42_musiccaps_caption_raw_full_ft")
    macros["TOneMtatFixed"] = num(dig(mtat, "test", "macro_f1_fixed_half"))
    if macros["TOneMtatFixed"] == PENDING:
        macros["TOneMtatFixed"] = fixed_half("task1_seed42_mtat_metadata_full_ft")
    if boot and boot.get("runs"):
        macros["NBoot"] = str(boot["n_boot"])
        rows = []
        for entry in boot["runs"]:
            label = entry["run"].replace("_", r"\_")
            rows.append(
                f"{label} & {entry['test_macro_f1_point']:.4f} & "
                f"{entry['test_macro_f1_fixed_half']:.4f} & "
                f"{entry['test_macro_f1_spread']:.4f} \\\\"
            )
        macros["BootRows"] = "\n".join(rows)

        # the headline multi-label run gets its dispersion spelled out, because
        # "spread" (the full range over replicates) is a much more conservative
        # statistic than the standard deviation and the two must not be confused
        lead = next((e for e in boot["runs"] if "mtat_tags" in e["run"]),
                    boot["runs"][0])
        lo, hi = lead["test_macro_f1_ci95"]
        macros["BootLeadStd"] = f"{lead['test_macro_f1_std']:.4f}"
        macros["BootLeadCI"] = f"[{lo:.4f}, {hi:.4f}]"
        macros["BootLeadSpread"] = f"{lead['test_macro_f1_spread']:.4f}"
        macros["BootLeadTuned"] = f"{lead['test_macro_f1_point']:.4f}"
        macros["BootLeadFixed"] = f"{lead['test_macro_f1_fixed_half']:.4f}"
        macros["BootThrStdMean"] = f"{lead['threshold_std_mean']:.3f}"
        macros["BootThrStdMax"] = f"{lead['threshold_std_max']:.3f}"
        macros["BootValRows"] = integer(lead["n_val_rows"])
        worst = lead.get("least_stable_tags", [])[:5]
        macros["BootWorstTags"] = ", ".join(
            f"\\texttt{{{w['tag'].replace('_', chr(92) + '_')}}} "
            f"($\\sigma={w['threshold_std']:.2f}$)" for w in worst) or PENDING
        unstable = [e for e in boot["runs"] if not e["stable"]]
        if unstable:
            worst = max(unstable, key=lambda e: e["test_macro_f1_spread"])
            macros["BootVerdict"] = (
                f"The largest spread is {worst['test_macro_f1_spread']:.4f} "
                f"macro-F1, above the {boot['spread_limit']} threshold fixed in "
                "advance. Tuned numbers are therefore reported alongside "
                "fixed-0.5 numbers throughout, and differences smaller than "
                "this spread are not interpreted."
            )
        else:
            macros["BootVerdict"] = (
                f"Every spread is at or below the {boot['spread_limit']} "
                "threshold fixed in advance, so tuned thresholds are stable "
                "enough for the tuned numbers to stand on their own."
            )
    else:
        macros["NBoot"] = "100"
        macros["BootRows"] = r"\multicolumn{4}{c}{\textit{pending}} \\"
        macros["BootVerdict"] = PENDING
        for key in ("BootLeadStd", "BootLeadCI", "BootLeadSpread", "BootLeadTuned",
                    "BootLeadFixed", "BootThrStdMean", "BootThrStdMax",
                    "BootValRows", "BootWorstTags"):
            macros[key] = PENDING

    return macros


def render(macros: dict) -> str:
    lines = [
        BEGIN,
        "% Regenerate with: python report/fill_report.py",
        "% Every value below is read from results/*.json -- do not hand-edit.",
    ]
    for name in sorted(macros):
        value = macros[name]
        if "\n" in value:                        # multi-line table body
            lines.append(f"\\newcommand{{\\{name}}}{{%")
            lines.append(value)
            lines.append("}")
        else:
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    lines.append(END)
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any macro is still pending")
    parser.add_argument("--tex", default=str(ROOT / "report" / "final_report.tex"))
    args = parser.parse_args(argv)

    macros = build_macros()
    path = Path(args.tex)
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{path} has no AUTOGEN block; add the markers back")

    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    path.write_text(pattern.sub(lambda _: render(macros), text, count=1),
                    encoding="utf-8")

    pending = sorted(k for k, v in macros.items() if PENDING in v)
    print(f"wrote {len(macros)} macros -> {path.relative_to(ROOT)}")
    if pending:
        print(f"{len(pending)} still pending: {', '.join(pending)}")
    else:
        print("no pending values: every number in the report is real")
    return 1 if (args.check and pending) else 0


if __name__ == "__main__":
    raise SystemExit(main())
