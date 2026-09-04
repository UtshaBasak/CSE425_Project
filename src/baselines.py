"""B1 (random / majority) and B4 (PCA + MLP) reference points.

B1 exists to make the headline numbers legible. On a 50-tag vocabulary with a
long-tailed prior, "macro-F1 0.31" means nothing until you know that predicting
tags at their marginal frequency already scores 0.08 -- and that a model
predicting *nothing* scores 0.00 macro-F1 while scoring 92% element accuracy,
which is exactly why accuracy is never reported here.

B4 is the "do you even need a graph?" control: the same 96-dim segment features,
mean-pooled over the track, PCA'd and fed to an MLP. If the GNN cannot beat it,
the topology is not doing any work.
"""
from __future__ import annotations

import numpy as np

from .metrics import (
    macro_f1,
    mean_auc_pr,
    micro_f1,
    per_tag_prf,
    tune_thresholds,
)
from .utils import get_logger

LOGGER = get_logger("gbmc.baselines")

__all__ = [
    "random_tag_baseline",
    "majority_tag_baseline",
    "pca_mlp_baseline",
    "prior_from_train",
]


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _observed(y: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & (y != -1)


def prior_from_train(y_train) -> np.ndarray:
    """Per-tag positive rate over the *observed* cells of the train split."""
    y = _as_2d(y_train)
    mask = _observed(y)
    counts = np.where(mask, y, 0.0).sum(axis=0)
    totals = mask.sum(axis=0)
    return np.divide(counts, np.maximum(totals, 1), out=np.zeros(y.shape[1]),
                     where=totals > 0)


def _score_block(y_true, y_score, thresholds, name: str, extra: dict | None = None) -> dict:
    """Common metric block. Note the absence of accuracy -- deliberate."""
    out = {
        "baseline": name,
        "macro_f1": macro_f1(y_true, y_score, thresholds),
        "micro_f1": micro_f1(y_true, y_score, thresholds),
        "mean_auc_pr": mean_auc_pr(y_true, y_score),
        "n_test": int(_as_2d(y_true).shape[0]),
        "n_tags": int(_as_2d(y_true).shape[1]),
    }
    out.update(extra or {})
    return out


def random_tag_baseline(y_train, y_test, seed: int = 42, n_repeats: int = 5) -> dict:
    """B1a: sample each tag independently at its train-split frequency.

    Averaged over ``n_repeats`` draws with the spread reported, because a single
    random draw on a 30-track test split has a standard deviation large enough
    to be mistaken for a result.
    """
    y_train = _as_2d(y_train)
    y_test = _as_2d(y_test)
    prior = prior_from_train(y_train)
    rng = np.random.default_rng(seed)

    runs = []
    for _ in range(int(n_repeats)):
        scores = rng.random(y_test.shape)
        # threshold at (1 - prior) so the expected positive rate matches train
        thresholds = 1.0 - prior
        runs.append(_score_block(y_test, scores, thresholds, "B1_random"))

    out = {"baseline": "B1_random", "n_repeats": int(n_repeats),
           "n_test": runs[0]["n_test"], "n_tags": runs[0]["n_tags"]}
    for key in ("macro_f1", "micro_f1", "mean_auc_pr"):
        values = [r[key] for r in runs]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["mean_train_prior"] = float(prior.mean())
    LOGGER.info("B1 random: macro-F1 %.4f +/- %.4f", out["macro_f1"], out["macro_f1_std"])
    return out


def majority_tag_baseline(y_train, y_test) -> dict:
    """B1b: predict each tag's majority class from train.

    For a long-tailed vocabulary this is the all-negative predictor for almost
    every tag, so macro-F1 collapses to ~0 while element accuracy is >90%. That
    contrast is the point: it is the single clearest argument in the report for
    why accuracy is not reported for multi-label tagging.
    """
    y_train = _as_2d(y_train)
    y_test = _as_2d(y_test)
    prior = prior_from_train(y_train)
    prediction = (prior >= 0.5).astype(np.float64)
    scores = np.tile(prediction, (y_test.shape[0], 1))

    mask = _observed(y_test)
    element_accuracy = float(
        np.mean((scores[mask] >= 0.5) == (y_test[mask] > 0.5)) if mask.any() else np.nan
    )
    out = _score_block(y_test, scores, np.full(y_test.shape[1], 0.5), "B1_majority",
                       {
                           "n_tags_predicted_positive": int(prediction.sum()),
                           # reported ONLY to demonstrate that accuracy is
                           # misleading here; never used as a headline metric.
                           "element_accuracy_do_not_report": element_accuracy,
                       })
    LOGGER.info(
        "B1 majority: macro-F1 %.4f while element accuracy is %.3f -- this is why "
        "accuracy is not a multi-label metric.", out["macro_f1"], element_accuracy,
    )
    return out


def pca_mlp_baseline(feats_train, y_train, feats_test, y_test, n_components: int = 128,
                     feats_val=None, y_val=None, seed: int = 42,
                     hidden: tuple[int, ...] = (256,), max_iter: int = 300) -> dict:
    """B4: PCA on mean-pooled segment features -> MLP -> per-tag sigmoids.

    PCA is fit on **train only** and applied to val/test, matching the
    normalisation rule used everywhere else. Thresholds are tuned on val when
    one is supplied, and otherwise on train -- never on test.
    """
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    Xtr = np.asarray(feats_train, dtype=np.float64)
    Xte = np.asarray(feats_test, dtype=np.float64)
    ytr = _as_2d(y_train)
    yte = _as_2d(y_test)

    # rows whose tags are entirely sentinel carry no supervision for this task
    train_rows = _observed(ytr).any(axis=1)
    Xtr, ytr = Xtr[train_rows], ytr[train_rows]
    if Xtr.shape[0] < 2:
        raise ValueError("PCA+MLP baseline needs at least two labelled train rows")

    scaler = StandardScaler().fit(Xtr)                    # train-only statistics
    n_components = int(min(n_components, Xtr.shape[0], Xtr.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed).fit(scaler.transform(Xtr))

    Ztr = pca.transform(scaler.transform(Xtr))
    Zte = pca.transform(scaler.transform(Xte))

    targets = np.where(_observed(ytr), ytr, 0.0)
    model = MLPClassifier(hidden_layer_sizes=tuple(hidden), max_iter=int(max_iter),
                          random_state=seed, early_stopping=False)
    model.fit(Ztr, targets)
    scores_test = _predict_scores(model, Zte, yte.shape[1])

    if feats_val is not None and y_val is not None:
        Zval = pca.transform(scaler.transform(np.asarray(feats_val, dtype=np.float64)))
        thresholds = tune_thresholds(_as_2d(y_val), _predict_scores(model, Zval, yte.shape[1]))
        threshold_source = "val"
    else:
        thresholds = tune_thresholds(ytr, _predict_scores(model, Ztr, yte.shape[1]))
        threshold_source = "train"

    out = _score_block(yte, scores_test, thresholds, "B4_pca_mlp", {
        "n_components": int(n_components),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "threshold_source": threshold_source,
        "n_train": int(Xtr.shape[0]),
    })
    out["per_tag"] = per_tag_prf(yte, scores_test, thresholds).to_dict("records")
    LOGGER.info("B4 PCA+MLP: macro-F1 %.4f (%d components, %.1f%% variance)",
                out["macro_f1"], n_components, 100 * out["explained_variance"])
    return out


def _predict_scores(model, Z: np.ndarray, n_tags: int) -> np.ndarray:
    """Coax a per-tag probability matrix out of sklearn's multi-label MLP."""
    proba = model.predict_proba(Z)
    if isinstance(proba, list):                            # one array per tag
        columns = [p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.reshape(-1)
                   for p in proba]
        scores = np.stack(columns, axis=1)
    else:
        scores = np.asarray(proba, dtype=np.float64)
    if scores.shape[1] != n_tags:
        # a tag that is constant in train is dropped by sklearn; re-insert it
        padded = np.zeros((scores.shape[0], n_tags), dtype=np.float64)
        padded[:, : scores.shape[1]] = scores[:, :n_tags]
        scores = padded
    return scores
