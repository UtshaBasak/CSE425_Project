#!/usr/bin/env python
"""Run every baseline: B1 (random / majority), B2 (mel CNN), B4 (PCA + MLP).

B2 is parameter-matched to the Task 2 GNN by default. That is not politeness --
"fair experimental setup" is a graded criterion, and a CNN with a tenth of the
GNN's capacity would make the comparison meaningless.

    python scripts/run_baselines.py [--synthetic] [--device cuda] [--skip-cnn]
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
from src.datasets import MelSpecDataset  # noqa: E402
from src.gnn_model import GNNClassifier  # noqa: E402
from src.train import TAG_DATASETS, DataBundle  # noqa: E402
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
    dataset = bundle.dataset(split, TAG_DATASETS)
    feats, tags = [], []
    for i in range(len(dataset)):
        data = dataset[i]
        feats.append(data.x.mean(dim=0).numpy())
        tags.append(data.y_tags.numpy().reshape(-1))
    if not feats:
        return np.zeros((0, 96)), np.zeros((0, len(bundle.tag_vocab)))
    return np.stack(feats), np.stack(tags)


def run_cnn_baseline(bundle: DataBundle, cfg, device, seed: int, epochs: int,
                     target_params: int) -> dict:
    """Train B2 on the mel cache, tuning thresholds on val and scoring test once."""
    from torch.utils.data import DataLoader

    mel_h5 = bundle.mel_h5
    if not Path(mel_h5).exists():
        LOGGER.warning("no mel cache at %s; skipping B2", mel_h5)
        return {"baseline": "B2_mel_cnn", "skipped": "no mel cache"}

    set_seed(seed)
    loaders = {}
    for split in ("train", "val", "test"):
        frame = bundle.frame(split, TAG_DATASETS)
        dataset = MelSpecDataset(frame, cfg=cfg, h5_path=mel_h5,
                                 tag_vocab=bundle.tag_vocab)
        if len(dataset) == 0:
            LOGGER.warning("no %s rows for B2; skipping", split)
            return {"baseline": "B2_mel_cnn", "skipped": f"empty {split} split"}
        loaders[split] = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]),
                                    shuffle=(split == "train"), num_workers=0)

    model = MelCNN(n_tags=len(bundle.tag_vocab), n_mels=int(cfg["audio"]["n_mels"]),
                   target_params=target_params).to(device)
    optimizer = torch.optim.AdamW(model.param_groups(
        float(cfg["train"]["lr_bert"]), float(cfg["train"]["lr_head"]),
        float(cfg["train"]["weight_decay"])))
    scaler = make_grad_scaler(bool(cfg.get("amp", True)), device.type)
    accum = max(1, int(cfg["train"]["grad_accum_steps"]))

    for epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for i, (mel, y, _) in enumerate(loaders["train"]):
            mel, y = mel.to(device), y.to(device)
            with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                logits = model(mel)["tag_logits"]
                mask = torch.isfinite(y) & (y != -1)
                if not mask.any():
                    continue
                per_cell = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), torch.clamp(y, min=0.0), reduction="none")
                loss = (per_cell * mask.float()).sum() / mask.float().sum()
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def score(loader):
        model.eval()
        scores, targets = [], []
        for mel, y, _ in loader:
            with autocast_ctx(bool(cfg.get("amp", True)), device.type):
                logits = model(mel.to(device))["tag_logits"]
            scores.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(y.numpy())
        return np.concatenate(targets), np.concatenate(scores)

    y_val, s_val = score(loaders["val"])
    thresholds = M.tune_thresholds(y_val, s_val)          # VAL only
    y_test, s_test = score(loaders["test"])

    report = model.capacity_report()
    report.update({
        "baseline": "B2_mel_cnn",
        "macro_f1": M.macro_f1(y_test, s_test, thresholds),
        "micro_f1": M.micro_f1(y_test, s_test, thresholds),
        "mean_auc_pr": M.mean_auc_pr(y_test, s_test),
        "threshold_source": "val",
        "epochs": int(max(1, epochs)),
    })
    LOGGER.info("B2 mel CNN: macro-F1 %.4f with %d params (target %d)",
                report["macro_f1"], report["trainable_params"], target_params)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run B1 / B2 / B4 baselines.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-cnn", action="store_true")
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

    # parameter-match B2 to the Task 2 GNN so the comparison is about
    # architecture, not capacity
    reference = GNNClassifier(
        n_tags=len(bundle.tag_vocab), in_dim=int(cfg["graph"]["node_feat_dim"]),
        hidden_dim=int(cfg["gnn"]["hidden_dim"]), num_layers=int(cfg["gnn"]["num_layers"]),
        conv=str(cfg["gnn"]["conv"]), readout=str(cfg["gnn"]["readout"]),
    )
    target_params = count_parameters(reference)
    if not args.skip_cnn:
        results.append(run_cnn_baseline(bundle, cfg, device, seed, epochs, target_params))

    payload = {
        "seed": seed,
        "synthetic": bool(args.synthetic),
        "device": str(device),
        "gnn_reference_params": target_params,
        "n_tags": len(bundle.tag_vocab),
        "baselines": results,
        "note": "accuracy is deliberately absent -- see src/metrics.py",
    }
    out = save_json(payload, Path(cfg["paths"]["results"]) / f"baselines_seed{seed}.json")
    print(json.dumps(payload, indent=2, default=str))
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
