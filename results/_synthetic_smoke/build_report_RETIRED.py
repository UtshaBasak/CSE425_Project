#!/usr/bin/env python
"""Render ``report/final_report.md`` plus every generated figure into one PDF.

Deliberately dependency-free (matplotlib only) -- pandoc and a LaTeX toolchain
are not available on the target machine, and a report build that needs a 4 GB
TeX install is a report build nobody runs.

    python report/build_report.py [--out report/final_report.pdf]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAGE = (8.27, 11.69)          # A4 portrait, inches
LEFT, TOP, BOTTOM = 0.08, 0.94, 0.06
LINE_HEIGHT = 0.0155


def _wrap(line: str, width: int = 96) -> list[str]:
    if not line.strip():
        return [""]
    indent = len(line) - len(line.lstrip())
    return textwrap.wrap(line, width=width, subsequent_indent=" " * (indent + 2)) or [""]


def _style(line: str) -> dict:
    stripped = line.lstrip()
    if stripped.startswith("### "):
        return {"size": 10.5, "weight": "bold", "text": stripped[4:], "space": 0.008}
    if stripped.startswith("## "):
        return {"size": 12, "weight": "bold", "text": stripped[3:], "space": 0.012}
    if stripped.startswith("# "):
        return {"size": 14, "weight": "bold", "text": stripped[2:], "space": 0.016}
    if stripped.startswith(">"):
        return {"size": 8, "weight": "normal", "text": line, "space": 0.0,
                "color": "#666666", "style": "italic"}
    if stripped.startswith("|") or stripped.startswith("```") or "  " in line[:4]:
        return {"size": 7.5, "weight": "normal", "text": line, "space": 0.0,
                "family": "monospace"}
    return {"size": 8.6, "weight": "normal", "text": line, "space": 0.0}


def text_pages(pdf: PdfPages, markdown: str) -> int:
    fig = ax = None
    y = 0.0
    pages = 0

    def new_page():
        nonlocal fig, ax, y, pages
        if fig is not None:
            pdf.savefig(fig)
            plt.close(fig)
        fig = plt.figure(figsize=PAGE)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = TOP
        pages += 1

    new_page()
    in_code = False
    for raw in markdown.splitlines():
        if raw.strip().startswith("```"):
            in_code = not in_code
        spec = _style(raw)
        width = 110 if spec.get("family") == "monospace" else 96
        chunks = [raw] if spec.get("family") == "monospace" else _wrap(spec["text"], width)
        y -= spec.get("space", 0.0)
        for chunk in chunks:
            if y < BOTTOM:
                new_page()
            ax.text(LEFT, y, chunk, fontsize=spec["size"],
                    fontweight=spec.get("weight", "normal"),
                    color=spec.get("color", "#111111"),
                    family=spec.get("family", "sans-serif"),
                    style=spec.get("style", "normal"),
                    va="top", ha="left", transform=ax.transAxes)
            y -= LINE_HEIGHT * (spec["size"] / 8.6)
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)
    return pages


def figure_pages(pdf: PdfPages, plots_dir: Path) -> list[str]:
    images = sorted(p for p in plots_dir.glob("*.png"))
    included = []
    for path in images:
        fig = plt.figure(figsize=PAGE)
        ax = fig.add_axes([0.06, 0.08, 0.88, 0.82])
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        fig.text(0.5, 0.955, path.stem.replace("_", " "), ha="center",
                 fontsize=12, fontweight="bold")
        fig.text(0.5, 0.035, f"results/plots/{path.name}", ha="center",
                 fontsize=7, color="#777777")
        pdf.savefig(fig)
        plt.close(fig)
        included.append(path.name)
    return included


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the report PDF.")
    parser.add_argument("--out", default=str(ROOT / "report" / "final_report.pdf"))
    parser.add_argument("--source", default=str(ROOT / "report" / "final_report.md"))
    parser.add_argument("--plots", default=str(ROOT / "results" / "plots"))
    args = parser.parse_args(argv)

    markdown = Path(args.source).read_text(encoding="utf-8")
    metrics_path = ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        markdown += (
            "\n\n---\n\n## Appendix: machine-readable results\n\n"
            f"Source: `results/metrics.json` ({'synthetic' if metrics.get('synthetic') else 'real'} data, "
            f"seed {metrics.get('seed')}, device {metrics.get('device')}).\n\n"
            f"Tasks with saved runs: {sorted(metrics.get('tasks', {}))}\n"
            f"Plot types written: {metrics.get('plot_types_written', [])}\n"
        )
    markdown += f"\n\nBuilt {date.today().isoformat()} by report/build_report.py\n"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        pages = text_pages(pdf, markdown)
        figures = figure_pages(pdf, Path(args.plots))
        info = pdf.infodict()
        info["Title"] = "GNN-BERT Music Context Understanding"
        info["Subject"] = "CSE425 project report (skeleton)"

    print(json.dumps({"pdf": str(out), "text_pages": pages,
                      "figure_pages": len(figures), "figures": figures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
