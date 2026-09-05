#!/usr/bin/env python
"""Run every baseline: B1 (random / majority), B2 (mel CNN), B4 (PCA + MLP).

    python scripts/run_baselines.py [--device cuda] [--skip-cnn]
                                    [--cnn-targets tags,genre]

**B2 is no longer parameter-matched to the GNN (A7.2).** It used to be, on the
theory that equal capacity made the comparison fair; what it actually produced
was 0.1654 macro-F1 on MTAT top-50 against a literature range of roughly
0.38-0.41, i.e. a broken baseline reported as a finding. Convolutional weights
are reused at every time-frequency position, so an equal parameter count buys
the two architectures wildly unequal amounts of computation, and matching it
starves the CNN. What is equalised now is the **compute budget** -- same GPU,
same epoch cap, same early-stopping rule -- and parameters and wall-clock are
reported for both models so the reader can see exactly what each one cost.

B2 also now trains on 3-second excerpts of the **native-resolution** mel cache
and averages per-chunk probabilities at inference, which is the standard MTAT
protocol, instead of consuming one 256-column summary of a whole 29-second clip.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import metrics as M  # noqa: E402
from src.baselines import majority_tag_baseline, pca_mlp_baseline, random_tag_baseline  # noqa: E402
from src.cnn_baseline import MelCNN  # noqa: E402
from src.datasets import ChunkedMelDataset  # noqa: E402
from src.gnn_model import GNNClassifier  # noqa: E402
from src.train import (TAG_DATASETS, DataBundle, corpora_for,  # noqa: E402
                       genre_names)
from src.utils import (  # noqa: E402
    autocast_ctx,
    count_parameters,
    get_device,
    get_logger,
    load_config,
    make_grad_scaler,
    parse_overrides,
    save_json,
    set_seed,
)

LOGGER = get_logger("gbmc.baselines.cli")


def _pooled_features(bundle: DataBundle, split: str):
    """Mean-pooled 96-dim segment features + the tag matrix for one split."""
    dataset = bundle.dataset(split, corpora_for(bundle.cfg, 'tag'))
    feats, tags = [], []
    for i in range(len(dataset)):
        data = dataset[i]
        feats.append(data.x.mean(dim=0).numpy())
        tags.append(data.y_tags.numpy().reshape(-1))
    if not feats:
        return np.zeros((0, 96)), np.zeros((0, len(bundle.tag_vocab)))
    return np.stack(feats), np.stack(tags)


def _full_mel_caches(cfg, corpora) -> dict:
    """``{corpus: mels_full_{corpus}.h5}`` for the caches that exist."""
    processed = Path(cfg["paths"]["processed"])
    found = {}
    for name in corpora:
        path = processed / f"mels_full_{name}.h5"
        if path.exists():
            found[name] = path
    return found


def run_cnn_baseline(bundle: DataBundle, cfg, device, seed: int, epochs: int,
                     target: str = "tags") -> dict:
    """Train B2 on short chunks of the full-resolution mel cache (A7.2).

    ``target="tags"`` is MTAT multi-label; ``target="genre"`` is FMA-small
    8-way. Thresholds (tags only) are tuned on validation and applied unchanged
    to test, exactly as every other model here does it.
    """
    import time

    from torch.utils.data import DataLoader

    from src.audio_features import compute_mel_stats

    corpora = corpora_for(cfg, "genre" if target == "genre" else "tag")
    caches = _full_mel_caches(cfg, corpora)
    if not caches:
        LOGGER.warning("no full-resolution mel cache for %s; run "
                       "scripts/build_mel_cache.py --datasets %s",
                       corpora, ",".join(corpora))
        return {"baseline": f"B2_mel_cnn_{target}", "skipped": "no mels_full cache"}

    set_seed(seed)
    chunk_s = float(cfg.get("baselines", {}).get("cnn_chunk_s", 3.0))
    n_eval_chunks = int(cfg.get("baselines", {}).get("cnn_eval_chunks", 9))
    batch = int(cfg.get("baselines", {}).get("cnn_batch_size", 32))
    # Inference stacks n_eval_chunks excerpts per clip, so the effective batch is
    # batch * n_eval_chunks: 32 x 9 measures 2.5 GB reserved on this card against
    # 707 MB for a training step. Sized separately rather than shared.
    eval_batch = int(cfg.get("baselines", {}).get("cnn_eval_batch_size", 8))
    workers = int(cfg.get("baselines", {}).get("cnn_workers", 3))

    # standardise on TRAIN statistics only, like every other model here
    mel_stats = None
    try:
        mel_stats = compute_mel_stats(bundle.frame("train", corpora), caches,
                                      split="train", max_tracks=2000)
        LOGGER.info("mel normalisation (train only): mean %.2f std %.2f over %d values",
                    mel_stats["mean"], mel_stats["std"], mel_stats["n_values"])
    except Exception as exc:                                   # noqa: BLE001
        LOGGER.warning("could not compute mel statistics (%s); B2 sees raw dB", exc)

    tag_vocab = [] if target == "genre" else bundle.tag_vocab
    names = genre_names(cfg, bundle.manifest[bundle.manifest["dataset"].isin(corpora)]) \
        if target == "genre" else []

    datasets, loaders = {}, {}
    for split in ("train", "val", "test"):
        frame = bundle.frame(split, corpora)
        if not len(frame):
            return {"baseline": f"B2_mel_cnn_{target}", "skipped": f"empty {split} split"}
        ds = ChunkedMelDataset(
            frame, cfg=cfg, h5_path=caches, tag_vocab=tag_vocab,
            chunk_s=chunk_s, mel_stats=mel_stats, seed=seed,
            mode="random" if split == "train" else "all",
            n_eval_chunks=n_eval_chunks,
        )
        datasets[split] = ds
        # gzip decompression dominates the input path (~2.2 ms/clip), so
        # workers matter here in a way they do not for the graph datasets
        loaders[split] = DataLoader(
            ds, batch_size=batch if split == "train" else eval_batch,
            shuffle=(split == "train"), num_workers=workers,
            persistent_workers=workers > 0)

    model = MelCNN(
        n_tags=0 if target == "genre" else len(bundle.tag_vocab),
        n_genres=len(names) if target == "genre" else 0,
        n_mels=int(cfg["audio"]["n_mels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(
        float(cfg["train"]["lr_bert"]), float(cfg["train"]["lr_head"]),
        float(cfg["train"]["weight_decay"])))
    amp = bool(cfg.get("amp", True))
    scaler = make_grad_scaler(amp, device.type)

    @torch.no_grad()
    def score(loader):
        model.eval()
        scores, targets, genre_logits, genre_true = [], [], [], []
        for mel, y, g, _ in loader:
            with autocast_ctx(amp, device.type):
                out = model(mel.to(device))
            if "tag_probs" in out:
                scores.append(out["tag_probs"].float().cpu().numpy())
                targets.append(y.numpy())
            if "genre_probs" in out:
                genre_logits.append(out["genre_probs"].float().cpu().numpy())
                genre_true.append(g.numpy())
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros((0,))   # noqa: E731
        return cat(targets), cat(scores), cat(genre_true), cat(genre_logits)

    metric_name = "genre_macro_f1" if target == "genre" else "macro_f1"
    patience = int(cfg["train"].get("early_stop_patience", 3))
    best, best_epoch, best_state, best_thresholds, stale = -np.inf, 0, None, None, 0
    history, started = [], time.time()

    # Accumulation is not decoration on a 4 GB card: it is what lets the
    # effective batch stay at 32 if the per-step batch has to drop for memory.
    accum = max(1, int(cfg["train"]["grad_accum_steps"])
                if cfg.get("baselines", {}).get("cnn_use_accumulation", False)
                else 1)

    for epoch in range(1, max(1, epochs) + 1):
        datasets["train"].set_epoch(epoch)      # a different excerpt each epoch
        model.train()
        epoch_start, total, n_batches = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (mel, y, g, _) in enumerate(loaders["train"]):
            mel = mel.to(device)
            with autocast_ctx(amp, device.type):
                out = model(mel)
                if target == "genre":
                    g = g.to(device).view(-1)
                    keep = g >= 0
                    if not keep.any():
                        continue
                    loss = torch.nn.functional.cross_entropy(
                        out["genre_logits"].float()[keep], g[keep])
                else:
                    y = y.to(device)
                    mask = torch.isfinite(y) & (y != -1)
                    if not mask.any():
                        continue
                    per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
                        out["tag_logits"].float(), torch.clamp(y, min=0.0),
                        reduction="none")
                    loss = (per_cell * mask.float()).sum() / mask.float().sum()
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().item())
            n_batches += 1

        y_val, s_val, gy_val, gp_val = score(loaders["val"])
        if target == "genre":
            val_metrics = M.multiclass_metrics(gy_val, gp_val, prefix="genre_")
            thresholds = None
        else:
            thresholds = M.tune_thresholds(y_val, s_val)          # VAL only
            val_metrics = {"macro_f1": M.macro_f1(y_val, s_val, thresholds)}
        value = float(val_metrics.get(metric_name, float("nan")))
        history.append({"epoch": epoch, "train_loss": total / max(n_batches, 1),
                        f"val_{metric_name}": value,
                        "seconds": round(time.time() - epoch_start, 1)})
        LOGGER.info("B2/%s epoch %d/%d | loss %.4f | val %s=%.4f | %.1fs",
                    target, epoch, epochs, total / max(n_batches, 1),
                    metric_name, value, history[-1]["seconds"])

        if np.isfinite(value) and value > best:
            best, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thresholds = thresholds
        else:
            stale += 1
            if stale >= patience:
                LOGGER.info("B2/%s early stop at epoch %d (best %d, %s=%.4f)",
                            target, epoch, best_epoch, metric_name, best)
                break

    wall_clock = round(time.time() - started, 1)
    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, s_test, gy_test, gp_test = score(loaders["test"])
    report = model.capacity_report()
    report.update({
        "baseline": f"B2_mel_cnn_{target}",
        "target": target,
        "corpora": list(corpora),
        "chunk_s": chunk_s,
        "chunk_frames": datasets["train"].chunk_frames,
        "frames_per_second": round(datasets["train"].frames_per_second, 2),
        "n_eval_chunks": n_eval_chunks,
        "batch_size": batch,
        "eval_batch_size": eval_batch,
        "dataloader_workers": workers,
        "grad_accum_steps": accum,
        "effective_batch": batch * accum,
        "mel_source": "mels_full (native resolution)",
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "wall_clock_s": wall_clock,
        "history": history,
        "mel_normalised": mel_stats is not None,
    })
    if target == "genre":
        report.update(M.multiclass_metrics(gy_test, gp_test, prefix="genre_"))
        report["threshold_source"] = "n/a (single-label)"
        LOGGER.info("B2 genre: accuracy %.4f macro-F1 %.4f | %s params | %.0fs",
                    report["genre_accuracy"], report["genre_macro_f1"],
                    f"{report['trainable_params']:,}", wall_clock)
    else:
        report.update({
            "macro_f1": M.macro_f1(y_test, s_test, best_thresholds),
            "micro_f1": M.micro_f1(y_test, s_test, best_thresholds),
            "mean_auc_pr": M.mean_auc_pr(y_test, s_test),
            "macro_f1_fixed_half": M.macro_f1(y_test, s_test, 0.5),
            "threshold_source": "val",
        })
        # A7.4 bootstraps thresholds from these, so keep the matrices
        scores_dir = Path(cfg["paths"]["results"]) / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores_dir / f"b2_seed{seed}_{target}_scores.npz",
                            val_y_true=y_val, val_y_score=s_val,
                            test_y_true=y_test, test_y_score=s_test,
                            tag_vocab=np.asarray(list(bundle.tag_vocab), dtype=object))
        LOGGER.info("B2 tags: macro-F1 %.4f (fixed-0.5 %.4f) | %s params | %.0fs",
                    report["macro_f1"], report["macro_f1_fixed_half"],
                    f"{report['trainable_params']:,}", wall_clock)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run B1 / B2 / B4 baselines.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-cnn", action="store_true")
    parser.add_argument("--cnn-targets", default="tags,genre",
                        help="which B2 runs to do: tags (MTAT), genre (FMA), or both")
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.override))
    if args.device:
        cfg["device"] = args.device
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    epochs = args.epochs if args.epochs is not None else int(cfg["train"]["epochs"])
    set_seed(seed)
    device = get_device(str(cfg.get("device", "cuda")))
    bundle = DataBundle(cfg, args.synthetic)

    Xtr, ytr = _pooled_features(bundle, "train")
    Xva, yva = _pooled_features(bundle, "val")
    Xte, yte = _pooled_features(bundle, "test")
    if ytr.size == 0 or yte.size == 0:
        LOGGER.error("no tag-bearing rows; nothing to baseline")
        return 1

    results = [random_tag_baseline(ytr, yte, seed=seed), majority_tag_baseline(ytr, yte)]
    try:
        b4 = pca_mlp_baseline(Xtr, ytr, Xte, yte, n_components=min(128, Xtr.shape[1]),
                              feats_val=Xva if Xva.size else None,
                              y_val=yva if yva.size else None, seed=seed)
        b4.pop("per_tag", None)
        results.append(b4)
    except Exception as exc:
        LOGGER.warning("B4 failed: %s", exc)

    # The GNN's parameter count is still recorded -- as context for the reader,
    # not as a constraint on B2. See the module docstring for why matching it
    # was the wrong call.
    reference = GNNClassifier(
        n_tags=len(bundle.tag_vocab), in_dim=int(cfg["graph"]["node_feat_dim"]),
        hidden_dim=int(cfg["gnn"]["hidden_dim"]), num_layers=int(cfg["gnn"]["num_layers"]),
        conv=str(cfg["gnn"]["conv"]), readout=str(cfg["gnn"]["readout"]),
    )
    gnn_params = count_parameters(reference)
    if not args.skip_cnn:
        for target in [t.strip() for t in args.cnn_targets.split(",") if t.strip()]:
            results.append(run_cnn_baseline(bundle, cfg, device, seed, epochs, target))

    payload = {
        "seed": seed,
        "synthetic": bool(args.synthetic),
        "device": str(device),
        "gnn_reference_params": gnn_params,
        "n_tags": len(bundle.tag_vocab),
        "baselines": results,
        "note": ("accuracy appears only for the single-label FMA genre runs; "
                 "for multi-label tagging it never does -- see src/metrics.py"),
    }
    out = save_json(payload, Path(cfg["paths"]["results"]) / f"baselines_seed{seed}.json")
    print(json.dumps(payload, indent=2, default=str))
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
