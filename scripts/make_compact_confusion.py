#!/usr/bin/env python
"""Compact confusion figure, sized for a single IEEE column.

    python scripts/make_compact_confusion.py

`plot_genre_confusion.py` renders one panel per run, which is four panels wide
(three GNN seeds plus B2). At 3563x822 that is 0.23 column-widths tall, so at
column width the class labels are illegible -- fine as a working artifact,
useless in the report.

This averages the GNN seeds into one panel and puts B2 beside it. Averaging is
the honest choice as well as the compact one: Table I reports the GNN over three
seeds, so a single-seed matrix beside a three-seed accuracy would invite the
reader to check one against the other and find them disagreeing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import ensure_dir, get_logger, load_config, project_root  # noqa: E402

LOGGER = get_logger("gbmc.confusion.compact")


def build(out_path: Path, cfg_path: str = "config.yaml"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = project_root()
    results = root / "results"
    cfg = load_config(cfg_path)
    names = list(cfg["data"]["genre_names"])

    # ---- GNN: mean over whatever seeds exist ----------------------------- #
    mats, accs = [], []
    for path in sorted(results.glob("task2_seed*_fma_genre.json")):
        test = json.loads(path.read_text(encoding="utf-8")).get("test", {})
        if "genre_confusion" not in test:
            continue
        mats.append(np.asarray(test["genre_confusion"], dtype=float))
        accs.append(float(test["genre_accuracy"]))
    if not mats:
        LOGGER.warning("no genre runs found")
        return None
    gnn = np.mean(mats, axis=0)
    gnn = gnn / np.clip(gnn.sum(axis=1, keepdims=True), 1e-9, None)

    # ---- B2 --------------------------------------------------------------- #
    b2 = None
    for path in sorted(results.glob("baselines_seed*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8")).get("baselines", []):
            if entry.get("baseline") == "B2_mel_cnn_genre" and "genre_confusion" in entry:
                b2 = np.asarray(entry["genre_confusion"], dtype=float)
                b2 = b2 / np.clip(b2.sum(axis=1, keepdims=True), 1e-9, None)
                b2_acc = float(entry.get("genre_accuracy", float("nan")))
    panels = [(f"(a) T2 GNN, mean of {len(mats)} seeds "
               f"(acc {np.mean(accs):.3f})", gnn)]
    if b2 is not None:
        panels.append((f"(b) B2 mel CNN (acc {b2_acc:.3f})", b2))

    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 3.1), squeeze=False)
    for ax, (title, mat) in zip(axes[0], panels):
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
        # Only the diagonal and the heavy confusions are worth printing; a full
        # 8x8 grid of numbers at this size is noise.
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                if v >= 0.10:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.4,
                            color="white" if v > 0.55 else "#222")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s (%d panels)", out_path, len(panels))
    return out_path


def main(argv=None) -> int:
    out = project_root() / "results" / "plots" / "genre_confusion_compact.png"
    build(out)
    print(f"compact confusion: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
