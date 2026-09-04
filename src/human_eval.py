"""Task 4 human evaluation: listening sheets and inter-rater agreement.

Two design decisions carry the whole exercise:

* **Randomised presentation order.** If every rater sees the same sequence,
  order effects and fatigue are confounded with item identity, and a shared
  drift looks like agreement.
* **Injected control pairs.** A fraction of items are deliberately mismatched
  (a caption paired with an unrelated clip). Raters who cannot separate controls
  from real retrievals were not listening -- and without controls, a table of
  4-out-of-5 ratings is indistinguishable from everyone defaulting to the middle
  of the scale. ``control_discrimination`` in :func:`compute_agreement` is the
  number that makes the rest of the human evaluation believable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, get_logger, resolve_path

LOGGER = get_logger("gbmc.humaneval")

__all__ = ["generate_listening_sheet", "compute_agreement", "SHEET_COLUMNS"]

SHEET_COLUMNS = [
    "item_id", "presentation_order", "query_track_id", "retrieved_track_id",
    "caption", "audio_path", "is_control", "true_rank", "rating",
]


def generate_listening_sheet(retrieval_results, n_items: int = 20, n_controls: int = 4,
                             seed: int = 42, out_path=None,
                             n_raters: int = 5) -> pd.DataFrame:
    """Build a rating sheet of ``n_items`` real pairs plus ``n_controls`` decoys.

    ``retrieval_results`` accepts the list produced by
    :func:`src.evaluate.export_retrieval_examples`, the dict written to
    ``retrieval_examples.json``, or a path to that file.

    Each rater gets an independently shuffled ``presentation_order``, so the
    sheet is written once and sliced per rater. The ``rating`` column is left
    blank for a human to fill on a 1-5 scale.
    """
    examples = _coerce_examples(retrieval_results)
    if not examples:
        raise ValueError("no retrieval examples to build a listening sheet from")

    rng = np.random.default_rng(seed)
    rows = []

    # ---- real pairs: query caption vs the top-1 retrieved clip -------------- #
    for i, example in enumerate(examples[: int(n_items)]):
        top = (example.get("top3") or [{}])[0]
        rows.append({
            "item_id": f"item_{i:03d}",
            "query_track_id": example.get("query_track_id", ""),
            "retrieved_track_id": top.get("track_id", ""),
            "caption": str(example.get("query_caption", ""))[:400],
            "audio_path": top.get("audio_path", ""),
            "is_control": False,
            "true_rank": example.get("true_rank"),
        })

    # ---- controls: a caption paired with a deliberately unrelated clip ------ #
    pool = [e for e in examples]
    for c in range(int(n_controls)):
        if len(pool) < 2:
            break
        a, b = rng.choice(len(pool), size=2, replace=False)
        rows.append({
            "item_id": f"control_{c:03d}",
            "query_track_id": pool[int(a)].get("query_track_id", ""),
            "retrieved_track_id": pool[int(b)].get("query_track_id", ""),
            "caption": str(pool[int(a)].get("query_caption", ""))[:400],
            "audio_path": "",
            "is_control": True,
            "true_rank": None,
        })

    frame = pd.DataFrame(rows)
    order = rng.permutation(len(frame))
    frame["presentation_order"] = np.argsort(order)
    frame = frame.sort_values("presentation_order").reset_index(drop=True)
    frame["rating"] = np.nan
    frame = frame[SHEET_COLUMNS]

    if out_path:
        out = ensure_dir(Path(resolve_path(out_path)).parent) / Path(out_path).name
        frame.to_csv(out, index=False)
        # one independently shuffled sheet per rater
        for r in range(int(n_raters)):
            per_rater = frame.sample(frac=1.0, random_state=seed + r).reset_index(drop=True)
            per_rater["presentation_order"] = np.arange(len(per_rater))
            per_rater.to_csv(out.parent / f"{out.stem}_rater{r + 1}.csv", index=False)
        LOGGER.info("wrote listening sheet (%d items, %d controls) + %d rater sheets to %s",
                    int(n_items), int(n_controls), int(n_raters), out.parent)
    return frame


def compute_agreement(ratings) -> dict:
    """Krippendorff alpha, pairwise Spearman, and control discrimination.

    ``ratings`` may be a wide DataFrame (rows = items, columns = raters), a long
    DataFrame with ``item_id``/``rater``/``rating``, or a 2-D array shaped
    ``[n_raters, n_items]``. An ``is_control`` column, when present, drives the
    discrimination check.

    Alpha is computed on the ordinal metric: a 1-5 Likert scale is ordered, and
    treating it as nominal (the default in several libraries) understates
    agreement by counting a 4-vs-5 disagreement as badly as 1-vs-5.
    """
    matrix, controls, item_ids = _coerce_ratings(ratings)
    n_raters, n_items = matrix.shape
    out: dict = {"n_raters": int(n_raters), "n_items": int(n_items)}
    if n_items == 0 or n_raters == 0:
        return out

    with np.errstate(invalid="ignore"):
        out["mean_rating"] = float(np.nanmean(matrix))
        out["std_rating"] = float(np.nanstd(matrix))

    # ---- Krippendorff alpha ------------------------------------------------ #
    try:
        import krippendorff

        out["krippendorff_alpha"] = float(
            krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
        )
    except Exception as exc:  # pragma: no cover - optional dependency / degenerate data
        LOGGER.warning("Krippendorff alpha unavailable: %s", exc)
        out["krippendorff_alpha"] = float("nan")

    # ---- pairwise Spearman ------------------------------------------------- #
    from scipy.stats import spearmanr

    rhos = []
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            mask = np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            if mask.sum() < 3:
                continue
            if np.all(matrix[i][mask] == matrix[i][mask][0]) or \
               np.all(matrix[j][mask] == matrix[j][mask][0]):
                continue                      # a constant rater has no rank order
            rho = spearmanr(matrix[i][mask], matrix[j][mask]).statistic
            if np.isfinite(rho):
                rhos.append(float(rho))
    out["pairwise_spearman_mean"] = float(np.mean(rhos)) if rhos else float("nan")
    out["pairwise_spearman_std"] = float(np.std(rhos)) if rhos else float("nan")
    out["n_rater_pairs"] = len(rhos)

    # ---- control discrimination -------------------------------------------- #
    if controls is not None and controls.any() and (~controls).any():
        real = matrix[:, ~controls]
        fake = matrix[:, controls]
        with np.errstate(invalid="ignore"):
            mean_real = float(np.nanmean(real))
            mean_control = float(np.nanmean(fake))
        out["mean_rating_real"] = mean_real
        out["mean_rating_control"] = mean_control
        out["control_discrimination"] = mean_real - mean_control
        try:
            from scipy.stats import mannwhitneyu

            stat = mannwhitneyu(real[np.isfinite(real)], fake[np.isfinite(fake)],
                                alternative="greater")
            out["control_discrimination_p"] = float(stat.pvalue)
        except Exception:  # pragma: no cover - degenerate input
            out["control_discrimination_p"] = float("nan")
        out["raters_discriminating"] = int(sum(
            np.nanmean(matrix[r, ~controls]) > np.nanmean(matrix[r, controls])
            for r in range(n_raters)
        ))
        if out["control_discrimination"] < 0.5:
            LOGGER.warning(
                "controls scored within %.2f of real pairs -- the ratings do not "
                "demonstrate that raters were discriminating, and the human "
                "evaluation should be reported as inconclusive.",
                out["control_discrimination"],
            )
    if item_ids is not None:
        out["item_ids"] = list(item_ids)
    return out


def _coerce_examples(source) -> list[dict]:
    if isinstance(source, (str, Path)):
        with open(resolve_path(source), "r", encoding="utf-8") as fh:
            source = json.load(fh)
    if isinstance(source, dict):
        source = source.get("examples", [])
    if isinstance(source, pd.DataFrame):
        return source.to_dict("records")
    return list(source or [])


def _coerce_ratings(ratings):
    """Normalise the accepted shapes to ``([n_raters, n_items], controls, ids)``."""
    controls = None
    item_ids = None

    if isinstance(ratings, (str, Path)):
        ratings = pd.read_csv(resolve_path(ratings))

    if isinstance(ratings, pd.DataFrame):
        frame = ratings.copy()
        if {"item_id", "rater", "rating"}.issubset(frame.columns):
            if "is_control" in frame.columns:
                flags = frame.drop_duplicates("item_id").set_index("item_id")["is_control"]
            else:
                flags = None
            wide = frame.pivot_table(index="rater", columns="item_id", values="rating")
            item_ids = list(wide.columns)
            controls = (
                np.asarray([bool(flags.get(i, False)) for i in item_ids]) if flags is not None
                else None
            )
            return wide.to_numpy(dtype=float), controls, item_ids

        if "is_control" in frame.columns:
            controls = frame["is_control"].astype(bool).to_numpy()
        if "item_id" in frame.columns:
            item_ids = frame["item_id"].tolist()
        rater_columns = [c for c in frame.columns
                         if c.lower().startswith("rater") or c.lower().startswith("rating_")]
        if not rater_columns:
            rater_columns = [c for c in frame.columns
                             if c not in SHEET_COLUMNS
                             and pd.api.types.is_numeric_dtype(frame[c])]
        matrix = frame[rater_columns].to_numpy(dtype=float).T
        return matrix, controls, item_ids

    matrix = np.asarray(ratings, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return matrix, controls, item_ids
