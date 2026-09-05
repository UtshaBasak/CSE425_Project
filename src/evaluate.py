"""One command regenerates every table and plot in the report.

    python -m src.evaluate [--synthetic] [--config config.yaml] [--tasks 1 2 3 4]

Writes ``results/metrics.json`` plus one file per plot type into
``results/plots/`` and ten retrieval examples into ``results/retrieval_examples/``.

The evaluation reads checkpoints where they exist and falls back to a short
in-process training run otherwise, so the command always produces a complete
artefact set rather than a half-filled report skeleton.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics as M  # noqa: E402
from .baselines import majority_tag_baseline, pca_mlp_baseline, random_tag_baseline  # noqa: E402
from .datasets import collate_texts, make_loader  # noqa: E402
from .fusion_model import FUSION_MODES  # noqa: E402
from .graph_builder import rewire_edges  # noqa: E402
from .train import (  # noqa: E402
    CAPTION_DATASETS,
    TAG_DATASETS,
    corpora_for,
    DataBundle,
    collect_scores,
    emotion_metrics,
    tagging_metrics,
)
from .utils import (  # noqa: E402
    SYNTHETIC,
    autocast_ctx,
    detect_provenance,
    guard_against_synthetic,
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_json,
    set_seed,
)

LOGGER = get_logger("gbmc.eval")

PLOT_TYPES = (
    "f1_vs_epoch", "per_tag_prf", "ablation", "tsne_genre", "tsne_mood",
    "retrieval", "graph_coherence", "seed_summary", "confusion_topk",
    "bert_attention", "case_study",
)

# A brand-neutral, colour-blind-safe categorical palette; light/dark agnostic.
PALETTE = ["#3b6ea5", "#c1666b", "#4f9d69", "#d4a03c", "#7a6ba8", "#48959b", "#b5762a"]


# --------------------------------------------------------------------------- #
# loading results
# --------------------------------------------------------------------------- #
def load_task_results(results_dir) -> dict[int, list[dict]]:
    """Group ``task{N}_seed{S}.json`` payloads by task."""
    out: dict[int, list[dict]] = {}
    for path in sorted(Path(resolve_path(results_dir)).glob("task*_seed*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError:
            LOGGER.warning("skipping unreadable result file %s", path)
            continue
        out.setdefault(int(payload.get("task", 0)), []).append(payload)
    return out


def _ensure_results(cfg, args) -> dict[int, list[dict]]:
    """Train anything missing so the report is never partially empty."""
    from .train import build_parser as train_parser
    from .train import main as train_main

    results = load_task_results(cfg["paths"]["results"])
    missing = [t for t in args.tasks if not results.get(t)]
    if not missing:
        return results
    if not args.train_missing:
        LOGGER.warning("no saved results for task(s) %s; pass --train-missing to "
                       "generate them", missing)
        return results

    for task in missing:
        LOGGER.info("no checkpoint for task %d; running a short training pass", task)
        argv = ["--task", str(task), "--config", args.config,
                "--device", args.device or str(cfg.get("device", "cpu")),
                "--override", f"train.epochs={args.fallback_epochs}"]
        if args.synthetic:
            argv.append("--synthetic")
        train_main(argv)
    return load_task_results(cfg["paths"]["results"])



def _guard_checkpoints(cfg, allow_synthetic: bool) -> None:
    """Refuse to build a report from checkpoints trained on synthetic data.

    Checked separately from the manifest: pointing at a real manifest while a
    stale synthetic checkpoint sits in ``results/checkpoints/`` is exactly the
    mix-up that produces convincing, fake figures.
    """
    ckpt_dir = resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints"))
    if not ckpt_dir.exists():
        return
    offenders = []
    for path in sorted(ckpt_dir.glob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # pragma: no cover - unreadable checkpoint
            continue
        if detect_provenance(payload) == SYNTHETIC or detect_provenance(path) == SYNTHETIC:
            offenders.append(path.name)
    if offenders:
        guard_against_synthetic(
            {"provenance": SYNTHETIC}, allow_synthetic,
            f"checkpoints {offenders[:4]} in {ckpt_dir}",
        )


def _load_compatible(model, state: dict, prefix: str = "") -> dict:
    """Load only the checkpoint tensors whose names *and shapes* both match.

    ``strict=False`` forgives missing keys but still raises on a shape mismatch,
    which is exactly the case that shows up here: a checkpoint trained with a
    different ``hidden_dim``, a different tag vocabulary, or a SAGE encoder being
    read into a GATv2 one for the attention case studies. Reporting how much was
    actually restored is more useful than crashing or silently loading nothing.
    """
    own = model.state_dict()
    loadable, skipped = {}, []
    for key, value in state.items():
        name = key[len(prefix):] if prefix and key.startswith(prefix) else key
        if prefix and not key.startswith(prefix):
            continue
        if name in own and own[name].shape == value.shape:
            loadable[name] = value
        else:
            skipped.append(name)
    if loadable:
        model.load_state_dict(loadable, strict=False)
    LOGGER.info("restored %d/%d tensors (%d incompatible)",
                len(loadable), len(own), len(skipped))
    return {"restored": len(loadable), "total": len(own), "skipped": len(skipped)}


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _finish(fig, ax_or_axes, out_path):
    fig.tight_layout()
    out = resolve_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    LOGGER.info("wrote %s", out)
    return out


def plot_f1_vs_epoch(results: dict[int, list[dict]], out_dir) -> Path | None:
    """Validation macro-F1 (or R@10 for Task 4) against epoch, per task."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for i, (task, runs) in enumerate(sorted(results.items())):
        for run in runs:
            history = run.get("history") or []
            if not history:
                continue
            key = next((k for k in ("val_macro_f1", "val_mean_R@10", "val_micro_f1")
                        if k in history[0]), None)
            if key is None:
                continue
            xs = [h["epoch"] for h in history]
            ys = [h.get(key, float("nan")) for h in history]
            ax.plot(xs, ys, marker="o", color=PALETTE[i % len(PALETTE)],
                    label=f"Task {task} (seed {run.get('seed')}) {key}", alpha=0.85)
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation metric")
    ax.set_title("Validation metric vs epoch")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    return _finish(fig, ax, Path(out_dir) / "f1_vs_epoch.png")


def plot_per_tag_prf(per_tag: pd.DataFrame, out_dir, tag_names=None, top_n: int = 25):
    """Per-tag precision/recall/F1 for the most-supported tags."""
    if per_tag is None or not len(per_tag):
        return None
    frame = per_tag.sort_values("support", ascending=False).head(top_n).iloc[::-1]
    labels = [
        tag_names[int(i)] if tag_names is not None and int(i) < len(tag_names) else f"tag {int(i)}"
        for i in frame["tag_index"]
    ]
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(frame))))
    ax.barh(y - 0.26, frame["precision"], height=0.25, color=PALETTE[0], label="precision")
    ax.barh(y, frame["recall"], height=0.25, color=PALETTE[1], label="recall")
    ax.barh(y + 0.26, frame["f1"], height=0.25, color=PALETTE[2], label="F1")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title(f"Per-tag precision / recall / F1 (top {len(frame)} by support)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    return _finish(fig, ax, Path(out_dir) / "per_tag_prf.png")


def plot_ablation(ablation: pd.DataFrame, out_dir):
    """Macro-F1 by fusion mode / architecture variant."""
    if ablation is None or not len(ablation):
        return None
    frame = ablation.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.42 * len(frame))))
    colours = [PALETTE[i % len(PALETTE)] for i in range(len(frame))]
    ax.barh(frame["variant"], frame["macro_f1"], color=colours)
    for i, (value, n) in enumerate(zip(frame["macro_f1"], frame.get("n_params", [None] * len(frame)))):
        label = f"{value:.3f}" + (f"  ({n/1e6:.1f}M)" if n else "")
        ax.text(value + 0.005, i, label, va="center", fontsize=8)
    ax.set_xlabel("test macro-F1")
    ax.set_title("Ablation: fusion mode and graph structure")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, max(0.05, float(frame["macro_f1"].max()) * 1.25))
    return _finish(fig, ax, Path(out_dir) / "ablation.png")


def plot_tsne(embeddings, labels, out_path, title: str, label_names=None,
              perplexity: int = 30, seed: int = 42) -> dict:
    """t-SNE scatter plus its quantitative companions (k-NN probe, silhouette).

    A t-SNE picture alone is not evidence -- the layout changes with perplexity
    and seed. The probe and silhouette printed in the title are what make the
    claim checkable.
    """
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    mask = labels != -1
    embeddings, labels = embeddings[mask], labels[mask]
    if embeddings.shape[0] < 5 or len(np.unique(labels)) < 2:
        LOGGER.warning("not enough labelled points for t-SNE %s", title)
        return {}

    perplexity = int(max(5, min(perplexity, (embeddings.shape[0] - 1) // 3)))
    coords = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                  init="pca", learning_rate="auto").fit_transform(embeddings)

    probe = M.knn_probe(embeddings, labels, k=10)
    sil = M.silhouette(embeddings, labels)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, value in enumerate(sorted(np.unique(labels))):
        sel = labels == value
        name = (label_names[int(value)] if label_names is not None
                and int(value) < len(label_names) else str(value))
        ax.scatter(coords[sel, 0], coords[sel, 1], s=22, alpha=0.8,
                   color=PALETTE[i % len(PALETTE)], label=name, edgecolors="none")
    ax.set_title(f"{title}\nperplexity={perplexity}  k-NN probe={probe:.3f}  "
                 f"silhouette={sil:.3f}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, markerscale=1.2)
    _finish(fig, ax, out_path)
    return {"knn_probe": probe, "silhouette": sil, "perplexity": perplexity,
            "n_points": int(embeddings.shape[0])}


def plot_retrieval(retrieval: dict, out_dir):
    """R@K bars in both directions, annotated with the gallery size."""
    if not retrieval:
        return None
    ks = [1, 5, 10]
    g2t = [retrieval.get(f"g2t_R@{k}", 0.0) for k in ks]
    t2g = [retrieval.get(f"t2g_R@{k}", 0.0) for k in ks]
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.19, g2t, width=0.36, color=PALETTE[0], label="audio graph -> caption")
    ax.bar(x + 0.19, t2g, width=0.36, color=PALETTE[1], label="caption -> audio graph")
    for xi, (a, b) in enumerate(zip(g2t, t2g)):
        ax.text(xi - 0.19, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"R@{k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("recall")
    ax.set_title(f"Task 4 retrieval (gallery size = {retrieval.get('gallery_size', '?')}, "
                 f"medR g2t = {retrieval.get('g2t_medR', float('nan')):.0f})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, ax, Path(out_dir) / "retrieval.png")


def plot_graph_coherence(scores: dict, out_dir):
    """S_graph for the trained model against its rewired control."""
    if not scores:
        return None
    keys = list(scores.keys())
    values = [scores[k] for k in keys]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(keys, values, color=[PALETTE[i % len(PALETTE)] for i in range(len(keys))])
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(r"$S_{graph}$")
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Graph coherence $S_{graph}$: fraction of edges with $\cos(h_i,h_j)>\tau$")
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, ax, Path(out_dir) / "graph_coherence.png")


def plot_seed_summary(summary: dict, out_dir):
    """Mean +/- std across seeds for the headline metrics."""
    keys = [k[:-5] for k in summary if k.endswith("_mean") and f"{k[:-5]}_std" in summary]
    keys = [k for k in keys if any(t in k for t in ("f1", "auc", "R@", "MRR", "r2", "mae"))]
    if not keys:
        return None
    means = [summary[f"{k}_mean"] for k in keys]
    stds = [summary[f"{k}_std"] for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(keys)), 4))
    ax.bar(keys, means, yerr=stds, capsize=4,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(keys))])
    ax.set_ylabel("value")
    ax.set_title(f"Seed-aggregated results (n={summary.get('n_seeds', '?')} seeds, mean +/- sd)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    return _finish(fig, ax, Path(out_dir) / "seed_summary.png")


def plot_confusion_topk(y_true, y_score, thresholds, out_dir, tag_names=None, top_n: int = 20):
    """Per-tag TP/FP/FN counts -- the multi-label stand-in for a confusion matrix.

    A square confusion matrix is undefined for multi-label prediction; this shows
    where the errors actually are without pretending the tags are exclusive.
    """
    frame = M.per_tag_prf(y_true, y_score, thresholds)
    if not len(frame):
        return None
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = (np.asarray(y_score, dtype=np.float64) >= np.asarray(thresholds).reshape(1, -1)
              if thresholds is not None else np.asarray(y_score) >= 0.5)
    observed = np.isfinite(y_true) & (y_true != -1)

    counts = []
    for j in range(y_true.shape[1]):
        m = observed[:, j]
        counts.append({
            "tag_index": j,
            "tp": int(np.sum((y_pred[m, j] == 1) & (y_true[m, j] == 1))),
            "fp": int(np.sum((y_pred[m, j] == 1) & (y_true[m, j] == 0))),
            "fn": int(np.sum((y_pred[m, j] == 0) & (y_true[m, j] == 1))),
        })
    frame = pd.DataFrame(counts).merge(frame[["tag_index", "support"]], on="tag_index")
    frame = frame.sort_values("support", ascending=False).head(top_n).iloc[::-1]
    labels = [
        tag_names[int(i)] if tag_names is not None and int(i) < len(tag_names) else f"tag {int(i)}"
        for i in frame["tag_index"]
    ]
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(frame))))
    ax.barh(y, frame["tp"], color=PALETTE[2], label="true positive")
    ax.barh(y, frame["fp"], left=frame["tp"], color=PALETTE[1], label="false positive")
    ax.barh(y, frame["fn"], left=frame["tp"] + frame["fp"], color=PALETTE[3], label="false negative")
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("count")
    ax.set_title("Per-tag error decomposition (multi-label: no square confusion matrix exists)")
    ax.legend(fontsize=8)
    return _finish(fig, ax, Path(out_dir) / "confusion_topk.png")


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def run_baselines(bundle: DataBundle, cfg) -> list[dict]:
    """B1 random / majority and B4 PCA+MLP on mean-pooled segment features."""
    import torch as _torch

    def _matrix(split: str):
        ds = bundle.dataset(split, corpora_for(cfg, 'tag'))
        feats, tags = [], []
        for i in range(len(ds)):
            data = ds[i]
            feats.append(data.x.mean(dim=0).numpy())
            tags.append(data.y_tags.numpy().reshape(-1))
        if not feats:
            return np.zeros((0, int(cfg["graph"]["node_feat_dim"]))), np.zeros((0, len(bundle.tag_vocab)))
        return np.stack(feats), np.stack(tags)

    Xtr, ytr = _matrix("train")
    Xva, yva = _matrix("val")
    Xte, yte = _matrix("test")
    if ytr.size == 0 or yte.size == 0:
        LOGGER.warning("no tag-bearing rows; skipping baselines")
        return []

    out = [random_tag_baseline(ytr, yte), majority_tag_baseline(ytr, yte)]
    try:
        b4 = pca_mlp_baseline(Xtr, ytr, Xte, yte,
                              n_components=min(128, Xtr.shape[1]),
                              feats_val=Xva if Xva.size else None,
                              y_val=yva if yva.size else None,
                              seed=int(cfg.get("seed", 42)))
        b4.pop("per_tag", None)
        out.append(b4)
    except Exception as exc:
        LOGGER.warning("B4 PCA+MLP failed: %s", exc)
    return out


def run_ablation(bundle: DataBundle, cfg, device, seed: int, epochs: int = 1) -> pd.DataFrame:
    """Train each fusion mode briefly and record test macro-F1, plus rewiring.

    The rewired-graph row is the control that distinguishes "the GNN uses
    musical structure" from "the GNN is a pooled-feature MLP with extra steps".
    """
    from .bert_encoder import BertTextEncoder, load_tokenizer
    from .fusion_model import GNNBertFusion, masked_multitask_loss
    from .gnn_model import GNNEncoder
    from .utils import count_parameters, make_grad_scaler

    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    rows = []
    variants = [(mode, False) for mode in FUSION_MODES] + [("cross_attention", True)]

    for mode, rewire in variants:
        set_seed(seed)
        gnn = GNNEncoder(
            in_dim=int(cfg["graph"]["node_feat_dim"]), hidden_dim=int(cfg["gnn"]["hidden_dim"]),
            num_layers=int(cfg["gnn"]["num_layers"]), conv=str(cfg["gnn"]["conv"]),
            dropout=float(cfg["gnn"]["dropout"]), readout=str(cfg["gnn"]["readout"]),
            residual=bool(cfg["gnn"]["residual"]),
        )
        bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="frozen_probe",
                               unfreeze_top_n=int(cfg["bert"]["unfreeze_top_n_layers"]))
        model = GNNBertFusion(gnn, bert, mode=mode, shared_dim=int(cfg["fusion"]["shared_dim"]),
                              n_heads=int(cfg["fusion"]["n_heads"]),
                              n_tags=len(bundle.tag_vocab), predict_emotion=True).to(device)

        train_loader = make_loader(bundle.dataset("train", corpora_for(cfg, "tag")), cfg, shuffle=True,
                                   seed=seed, num_workers=0)
        val_loader = make_loader(bundle.dataset("val", corpora_for(cfg, "tag")), cfg, shuffle=False,
                                 seed=seed, num_workers=0)
        test_loader = make_loader(bundle.dataset("test", corpora_for(cfg, "tag")), cfg, shuffle=False,
                                  seed=seed, num_workers=0)

        optimizer = torch.optim.AdamW(model.param_groups(
            float(cfg["train"]["lr_bert"]), float(cfg["train"]["lr_head"]),
            float(cfg["train"]["weight_decay"])))
        scaler = make_grad_scaler(bool(cfg.get("amp", True)), device.type)
        accum = max(1, int(cfg["train"]["grad_accum_steps"]))

        model.train()
        for _ in range(max(1, epochs)):
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(train_loader):
                if rewire:
                    batch = _rewire_batch(batch, seed)
                batch = batch.to(device)
                ids, mask = collate_texts(batch, tokenizer, int(cfg["bert"]["max_length"]), device)
                with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                    out = model(batch, ids, mask)
                    loss, _ = masked_multitask_loss(out, batch, cfg)
                if not torch.isfinite(loss):
                    continue
                scaler.scale(loss / accum).backward()
                if (i + 1) % accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        val = collect_scores(model, val_loader, device, cfg, 3, tokenizer)
        thresholds = M.tune_thresholds(val["targets"], val["scores"])   # VAL only
        test = collect_scores(model, test_loader, device, cfg, 3, tokenizer)
        scores = tagging_metrics(test, thresholds)
        rows.append({
            "variant": f"{mode}{' + rewired graph' if rewire else ''}",
            "mode": mode,
            "rewired": bool(rewire),
            "macro_f1": scores["macro_f1"],
            "micro_f1": scores["micro_f1"],
            "mean_auc_pr": scores.get("mean_auc_pr", float("nan")),
            "n_params": count_parameters(model),
        })
        LOGGER.info("ablation %-28s macro-F1 %.4f", rows[-1]["variant"], rows[-1]["macro_f1"])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _rewire_batch(batch, seed: int):
    """Apply degree-preserving rewiring to every graph in a PyG batch."""
    from torch_geometric.data import Batch

    graphs = batch.to_data_list()
    return Batch.from_data_list([rewire_edges(g, preserve_degree=True, seed=seed + i)
                                 for i, g in enumerate(graphs)])


def compute_graph_coherence(bundle: DataBundle, cfg, device, seed: int) -> dict:
    """S_graph on a trained encoder's node states, versus the rewired control."""
    from .gnn_model import GNNEncoder

    set_seed(seed)
    encoder = GNNEncoder(
        in_dim=int(cfg["graph"]["node_feat_dim"]), hidden_dim=int(cfg["gnn"]["hidden_dim"]),
        num_layers=int(cfg["gnn"]["num_layers"]), conv=str(cfg["gnn"]["conv"]),
        dropout=0.0, readout=str(cfg["gnn"]["readout"]),
        residual=bool(cfg["gnn"]["residual"]),
    ).to(device).eval()

    ckpt = resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints"))
    for name in (f"task3_seed{seed}_best.pt", f"task2_seed{seed}_best.pt"):
        path = ckpt / name
        if path.exists():
            state = torch.load(path, map_location=device, weights_only=False)["model_state"]
            prefix = "gnn." if name.startswith("task3") else "encoder."
            report = _load_compatible(encoder, state, prefix=prefix)
            if report["restored"]:
                LOGGER.info("loaded GNN weights from %s", path.name)
                break

    tau = float(cfg["eval"].get("graph_coherence_tau", 0.5))
    loader = make_loader(bundle.dataset("test"), cfg, shuffle=False, seed=seed, num_workers=0)
    real, rewired, cosines = [], [], []
    with torch.no_grad():
        for batch in loader:
            control = _rewire_batch(batch, seed).to(device)
            batch = batch.to(device)
            _, h = encoder(batch, return_nodes=True)
            real.append(M.graph_coherence_score(h, batch.edge_index, tau))
            cosines.append(_mean_edge_cosine(h, batch.edge_index))
            _, hc = encoder(control, return_nodes=True)
            rewired.append(M.graph_coherence_score(hc, control.edge_index, tau))
    finite = lambda xs: float(np.nanmean(xs)) if xs else float("nan")  # noqa: E731
    out = {
        "S_graph_real": finite(real),
        "S_graph_rewired": finite(rewired),
        "tau": tau,
        # S_graph is an indicator above tau, so it saturates at 1.0 once the
        # encoder oversmooths. The raw mean cosine says whether a 1.0 means
        # "structure is coherent" or "every node embedding is the same vector".
        "mean_edge_cosine": finite(cosines),
    }
    if np.isfinite(out["mean_edge_cosine"]) and out["mean_edge_cosine"] > 0.99:
        LOGGER.warning(
            "mean edge cosine is %.4f: node states have collapsed, so S_graph=%.3f "
            "reflects oversmoothing rather than musical coherence. Reduce "
            "gnn.num_layers or train longer before reporting it.",
            out["mean_edge_cosine"], out["S_graph_real"],
        )
    return out


def _mean_edge_cosine(h, edge_index) -> float:
    """Mean cos(h_i, h_j) over non-self edges -- the diagnostic behind S_graph."""
    h = np.asarray(h.detach().cpu().numpy(), dtype=np.float64)
    ei = edge_index.detach().cpu().numpy()
    keep = ei[0] != ei[1]
    if not keep.any():
        return float("nan")
    src, dst = ei[0][keep], ei[1][keep]
    unit = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(unit[src] * unit[dst], axis=1)))


def export_retrieval_examples(bundle: DataBundle, cfg, device, seed: int,
                              out_dir, n_examples: int = 10) -> list[dict]:
    """Ten qualitative retrieval examples, at least two of them failures.

    Failure cases are chosen deliberately -- a qualitative section made only of
    successes is not evidence, and the rubric asks for honest error analysis.
    """
    from .bert_encoder import BertTextEncoder, load_tokenizer
    from .contrastive import DualEncoder, build_similarity_matrix
    from .gnn_model import GNNEncoder

    out = ensure_dir(out_dir)
    set_seed(seed)
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                     hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                     num_layers=int(cfg["gnn"]["num_layers"]), conv=str(cfg["gnn"]["conv"]),
                     dropout=0.0, readout=str(cfg["gnn"]["readout"]))
    bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="frozen_probe")
    model = DualEncoder(gnn, bert, shared_dim=int(cfg["fusion"]["shared_dim"]),
                        temperature_init=float(cfg["contrastive"]["temperature_init"])).to(device)

    ckpt = resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")) / \
        f"task4_seed{seed}_best.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        _load_compatible(model, payload["model_state"])
        LOGGER.info("loaded Task 4 weights from %s", ckpt.name)
    model.eval()

    loader = make_loader(bundle.dataset("test", corpora_for(cfg, "caption")), cfg, shuffle=False,
                         seed=seed, num_workers=0, batch_size=16)
    graphs, texts, ids, captions = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            tok_ids, mask = collate_texts(batch, tokenizer, int(cfg["bert"]["max_length"]), device)
            with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                graphs.append(model.encode_graph(batch).float().cpu())
                texts.append(model.encode_text(tok_ids, mask).float().cpu())
            ids += list(batch.track_id) if isinstance(batch.track_id, list) else [batch.track_id]
            captions += list(batch.text) if isinstance(batch.text, list) else [batch.text]

    if not graphs:
        LOGGER.warning("no caption-bearing test rows; skipping retrieval examples")
        return []

    sim = build_similarity_matrix(torch.cat(graphs), torch.cat(texts))
    ranks = (sim > sim[np.arange(len(sim)), np.arange(len(sim))][:, None]).sum(axis=1) + 1

    successes = list(np.argsort(ranks)[: max(1, n_examples - 2)])
    failures = list(np.argsort(-ranks)[:2])                  # the two worst queries
    chosen = successes + [i for i in failures if i not in successes]

    examples = []
    for i in chosen[:n_examples]:
        order = np.argsort(-sim[i])[:3]
        examples.append({
            "query_track_id": ids[int(i)],
            "query_caption": captions[int(i)][:400],
            "true_rank": int(ranks[int(i)]),
            "is_failure": bool(ranks[int(i)] > 5),
            "top3": [
                {"rank": r + 1, "track_id": ids[int(j)], "caption": captions[int(j)][:300],
                 "score": float(sim[i, j]), "is_correct": bool(int(j) == int(i))}
                for r, j in enumerate(order)
            ],
        })
    save_json({"n_examples": len(examples),
               "n_failures": sum(e["is_failure"] for e in examples),
               "gallery_size": int(sim.shape[1]),
               "examples": examples}, out / "retrieval_examples.json")
    for k, example in enumerate(examples):
        with open(out / f"example_{k:02d}.md", "w", encoding="utf-8") as fh:
            fh.write(f"# Retrieval example {k} "
                     f"({'FAILURE' if example['is_failure'] else 'success'})\n\n")
            fh.write(f"**Query track**: `{example['query_track_id']}`  \n")
            fh.write(f"**Caption**: {example['query_caption']}\n\n")
            fh.write(f"**Rank of the true match**: {example['true_rank']} "
                     f"of {sim.shape[1]}\n\n## Top 3 retrieved\n\n")
            for hit in example["top3"]:
                mark = "OK" if hit["is_correct"] else "--"
                fh.write(f"{hit['rank']}. [{mark}] `{hit['track_id']}` "
                         f"(score {hit['score']:.3f})\n\n    {hit['caption']}\n\n")
    LOGGER.info("wrote %d retrieval examples (%d failures) to %s",
                len(examples), sum(e["is_failure"] for e in examples), out)
    return examples


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate every table and plot.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tasks", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--train-missing", action="store_true",
                        help="train any task that has no saved result")
    parser.add_argument("--fallback-epochs", type=int, default=1)
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--ablation-epochs", type=int, default=1)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    if args.device:
        cfg["device"] = args.device
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(str(cfg.get("device", "cuda")))
    plots_dir = ensure_dir(Path(cfg["paths"]["results"]) / "plots")
    examples_dir = ensure_dir(Path(cfg["paths"]["results"]) / "retrieval_examples")

    bundle = DataBundle(cfg, args.synthetic)
    # A0.1 guard: synthetic data must never reach a reported table or figure.
    guard_against_synthetic(bundle.manifest, args.synthetic, "the evaluation manifest")
    results = _ensure_results(cfg, args)
    _guard_checkpoints(cfg, args.synthetic)
    written_plots: dict[str, str] = {}
    metrics: dict = {
        "config_path": str(cfg.get("_config_path", args.config)),
        "synthetic": bool(args.synthetic),
        "device": str(device),
        "seed": seed,
        "n_tags": len(bundle.tag_vocab),
        "tag_vocab": list(bundle.tag_vocab),
        "split_sizes": bundle.manifest["split"].value_counts().to_dict(),
    }

    # ---- per-task summaries + seed aggregation ---------------------------- #
    metrics["tasks"] = {}
    for task, runs in sorted(results.items()):
        flat = []
        for run in runs:
            row = {"seed": run.get("seed")}
            row.update({k: v for k, v in (run.get("test") or {}).items()
                        if isinstance(v, (int, float))})
            flat.append(row)
        metrics["tasks"][f"task{task}"] = {
            "n_runs": len(runs),
            "per_seed": flat,
            "aggregated": M.aggregate_seeds([{k: v for k, v in row.items() if k != "seed"}
                                             for row in flat]),
            "best_val_metric": [run.get("best_val_metric") for run in runs],
            "threshold_source": sorted({run.get("threshold_source", "val") for run in runs}),
        }
    path = plot_f1_vs_epoch(results, plots_dir)
    if path:
        written_plots["f1_vs_epoch"] = str(path)

    # ---- per-tag table + error decomposition from the best tagging model --- #
    tagging_task = next((t for t in (3, 2, 1) if results.get(t)), None)
    if tagging_task:
        run = results[tagging_task][0]
        thresholds = run.get("thresholds")
        model_scores = _rescore_from_checkpoint(bundle, cfg, device, tagging_task, run)
        if model_scores is not None:
            y_true, y_score = model_scores
            frame = M.per_tag_prf(y_true, y_score, thresholds)
            frame.to_csv(Path(cfg["paths"]["results"]) / "per_tag_prf.csv", index=False)
            metrics["per_tag_prf"] = frame.to_dict("records")
            p = plot_per_tag_prf(frame, plots_dir, bundle.tag_vocab)
            if p:
                written_plots["per_tag_prf"] = str(p)
            p = plot_confusion_topk(y_true, y_score, thresholds, plots_dir, bundle.tag_vocab)
            if p:
                written_plots["confusion_topk"] = str(p)

    # ---- baselines --------------------------------------------------------- #
    metrics["baselines"] = run_baselines(bundle, cfg)

    # ---- ablation ---------------------------------------------------------- #
    if not args.skip_ablation:
        ablation = run_ablation(bundle, cfg, device, seed, epochs=args.ablation_epochs)
        ablation.to_csv(Path(cfg["paths"]["results"]) / "ablation.csv", index=False)
        metrics["ablation"] = ablation.to_dict("records")
        p = plot_ablation(ablation, plots_dir)
        if p:
            written_plots["ablation"] = str(p)

    # ---- embeddings, t-SNE, probes ----------------------------------------- #
    embeddings, genres, moods = _collect_embeddings(bundle, cfg, device, seed)
    if embeddings.size:
        stats = plot_tsne(embeddings, genres, plots_dir / "tsne_genre.png",
                          "Fused representation coloured by genre",
                          label_names=bundle.genres or None,
                          perplexity=int(cfg["eval"]["tsne_perplexity"]), seed=seed)
        if stats:
            metrics["tsne_genre"] = stats
            written_plots["tsne_genre"] = str(plots_dir / "tsne_genre.png")
        stats = plot_tsne(embeddings, moods, plots_dir / "tsne_mood.png",
                          "Fused representation coloured by mood quadrant",
                          label_names=["low V / low A", "low V / high A",
                                       "high V / low A", "high V / high A"],
                          perplexity=int(cfg["eval"]["tsne_perplexity"]), seed=seed)
        if stats:
            metrics["tsne_mood"] = stats
            written_plots["tsne_mood"] = str(plots_dir / "tsne_mood.png")

    # ---- retrieval --------------------------------------------------------- #
    retrieval = {}
    if results.get(4):
        retrieval = {k: v for k, v in (results[4][0].get("test") or {}).items()}
    if retrieval:
        metrics["retrieval"] = retrieval
        p = plot_retrieval(retrieval, plots_dir)
        if p:
            written_plots["retrieval"] = str(p)
    metrics["retrieval_examples"] = export_retrieval_examples(
        bundle, cfg, device, seed, examples_dir, int(cfg["eval"].get("retrieval_examples", 10))
    )

    # ---- S_graph ----------------------------------------------------------- #
    coherence = compute_graph_coherence(bundle, cfg, device, seed)
    metrics["graph_coherence"] = coherence
    p = plot_graph_coherence({"trained graph": coherence["S_graph_real"],
                              "rewired control": coherence["S_graph_rewired"]}, plots_dir)
    if p:
        written_plots["graph_coherence"] = str(p)

    # ---- attention figures and case studies -------------------------------- #
    try:
        from .attention_viz import generate_bert_examples, generate_case_studies

        metrics["bert_attention_examples"] = generate_bert_examples(
            bundle, cfg, device, seed, plots_dir, n_examples=5
        )
        if metrics["bert_attention_examples"]:
            written_plots["bert_attention"] = metrics["bert_attention_examples"][0]["plot"]
        metrics["case_studies"] = generate_case_studies(
            bundle, cfg, device, seed, plots_dir, n_cases=3
        )
        if metrics["case_studies"]:
            written_plots["case_study"] = metrics["case_studies"][0]["graph_plot"]
    except Exception as exc:
        LOGGER.warning("attention figures unavailable: %s", exc)
        metrics.setdefault("bert_attention_examples", [])
        metrics.setdefault("case_studies", [])

    # ---- seed aggregation plot --------------------------------------------- #
    all_runs = [dict(seed=r.get("seed"), **{k: v for k, v in (r.get("test") or {}).items()
                                            if isinstance(v, (int, float))})
                for runs in results.values() for r in runs]
    summary = M.aggregate_seeds([{k: v for k, v in r.items() if k != "seed"} for r in all_runs])
    metrics["seed_summary"] = summary
    p = plot_seed_summary(summary, plots_dir)
    if p:
        written_plots["seed_summary"] = str(p)

    # ---- MusicCaps survival, if the log exists ----------------------------- #
    log_path = resolve_path(cfg["paths"]["splits"]) / "musiccaps_download_log.csv"
    if log_path.exists():
        from .splits import musiccaps_survival_summary

        metrics["musiccaps_survival"] = musiccaps_survival_summary(pd.read_csv(log_path))

    metrics["plots"] = written_plots
    metrics["plot_types_written"] = sorted(written_plots)
    out = save_json(metrics, Path(cfg["paths"]["results"]) / "metrics.json")
    LOGGER.info("wrote %s with %d plot types", out, len(written_plots))
    print(json.dumps({"metrics_json": str(out), "plots": written_plots}, indent=2))
    return 0


def _rescore_from_checkpoint(bundle, cfg, device, task: int, run: dict):
    """Re-run the saved model over test to get raw scores for the tables."""
    from .bert_encoder import BertTagClassifier, BertTextEncoder, load_tokenizer
    from .fusion_model import GNNBertFusion
    from .gnn_model import GNNClassifier, GNNEncoder

    ckpt = resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")) / \
        f"task{task}_seed{run.get('seed', cfg.get('seed', 42))}_best.pt"
    tokenizer = load_tokenizer(cfg["bert"]["model_name"]) if task in (1, 3) else None
    n_tags = len(bundle.tag_vocab)

    if task == 1:
        model = BertTagClassifier(n_tags, cfg["bert"]["model_name"], freeze_mode="frozen_probe")
    elif task == 2:
        model = GNNClassifier(n_tags, in_dim=int(cfg["graph"]["node_feat_dim"]),
                              hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                              num_layers=int(cfg["gnn"]["num_layers"]),
                              conv=str(cfg["gnn"]["conv"]), dropout=0.0,
                              readout=str(cfg["gnn"]["readout"]))
    else:
        gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                         hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                         num_layers=int(cfg["gnn"]["num_layers"]),
                         conv=str(cfg["gnn"]["conv"]), dropout=0.0,
                         readout=str(cfg["gnn"]["readout"]))
        bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="frozen_probe")
        model = GNNBertFusion(gnn, bert, mode=str(cfg["fusion"]["mode"]),
                              shared_dim=int(cfg["fusion"]["shared_dim"]),
                              n_heads=int(cfg["fusion"]["n_heads"]), n_tags=n_tags)
    model = model.to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        _load_compatible(model, payload["model_state"])
    else:
        LOGGER.warning("no checkpoint at %s; scoring an untrained model", ckpt)

    loader = make_loader(bundle.dataset("test", corpora_for(cfg, "tag")), cfg, shuffle=False,
                         seed=int(cfg.get("seed", 42)), num_workers=0)
    if len(loader.dataset) == 0:
        return None
    collected = collect_scores(model, loader, device, cfg, task, tokenizer)
    return collected["targets"], collected["scores"]


def _collect_embeddings(bundle, cfg, device, seed: int):
    """Fused embeddings for the whole test split, plus genre and mood labels."""
    from .bert_encoder import BertTextEncoder, load_tokenizer
    from .fusion_model import GNNBertFusion
    from .gnn_model import GNNEncoder

    set_seed(seed)
    tokenizer = load_tokenizer(cfg["bert"]["model_name"])
    gnn = GNNEncoder(in_dim=int(cfg["graph"]["node_feat_dim"]),
                     hidden_dim=int(cfg["gnn"]["hidden_dim"]),
                     num_layers=int(cfg["gnn"]["num_layers"]), conv=str(cfg["gnn"]["conv"]),
                     dropout=0.0, readout=str(cfg["gnn"]["readout"]))
    bert = BertTextEncoder(cfg["bert"]["model_name"], freeze_mode="frozen_probe")
    model = GNNBertFusion(gnn, bert, mode=str(cfg["fusion"]["mode"]),
                          shared_dim=int(cfg["fusion"]["shared_dim"]),
                          n_heads=int(cfg["fusion"]["n_heads"]),
                          n_tags=len(bundle.tag_vocab)).to(device)
    ckpt = resolve_path(cfg["paths"].get("checkpoints", "results/checkpoints")) / \
        f"task3_seed{seed}_best.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        _load_compatible(model, payload["model_state"])
    model.eval()

    loader = make_loader(bundle.dataset("test"), cfg, shuffle=False, seed=seed, num_workers=0)
    embeddings, genres, moods = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ids, mask = collate_texts(batch, tokenizer, int(cfg["bert"]["max_length"]), device)
            with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                out = model(batch, ids, mask)
            embeddings.append(out["z"].float().cpu().numpy())
            genre = getattr(batch, "true_genre", None)
            genres.append((genre if genre is not None else batch.y_genre).view(-1).cpu().numpy())
            valence = batch.y_valence.view(-1).cpu().numpy()
            arousal = batch.y_arousal.view(-1).cpu().numpy()
            quadrant = np.where(
                np.isfinite(valence) & np.isfinite(arousal),
                (valence >= 5).astype(int) * 2 + (arousal >= 5).astype(int),
                -1,
            )
            moods.append(quadrant)
    if not embeddings:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    return (np.concatenate(embeddings), np.concatenate(genres).astype(int),
            np.concatenate(moods).astype(int))


if __name__ == "__main__":
    raise SystemExit(main())
