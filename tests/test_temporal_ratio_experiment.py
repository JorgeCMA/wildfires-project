"""Opt-in experiment: class-balance (fire:background) x temporal split.

This file does **not** touch the existing pipeline; it only recombines the
public units (``occurrence`` + ``occurrence_model``) to answer two questions:

1. How does the occurrence model rank when the background pool is subsampled
   to *different negative:positive ratios* (including ratios below 1, i.e. a
   deliberately smaller sample)?
2. How does it generalise when **2023 is the training year and 2024 the
   validation year** (a strictly temporal, no-shuffle split)?

Design notes
------------
- Negatives are drawn *per year* so the requested ratio holds inside both the
  training and the validation partition; when a year's pool is smaller than
  the requested count, the whole pool is used (same fallback as
  ``sample_background``).
- The split is temporal, so there is a single evaluation instead of the
  blocked leave-one-group-out folds used by notebook 04. The one-hot encoding
  and the scaler are still fit on the 2023 train portion only. The PR-AUC
  reference is the 2024 fold prevalence.
- Run with ``pytest -m experiment -s`` (excluded from the default suite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from sklearn.base import clone  # type: ignore[import-untyped]

from wildfire.analysis.occurrence_model import (
    categorical_columns,
    categorical_to_dummies,
    continuous_columns,
    fold_scaler_apply,
    fold_scaler_fit,
    model_candidates,
)
from wildfire.data.io import load_csv
from wildfire.processing.occurrence import (
    BACKGROUND_ID_COL,
    EVERYDAY_FEATURES,
    LABEL_COL,
    assemble_modeling_frame,
    load_background_days,
    occurrence_features,
    rank_metrics,
    validate_background_days,
)

pytestmark = pytest.mark.experiment

POSITIVES_CSV = "data/processed/expanded/firms_spain_final_lags_index.csv"

TRAIN_YEAR = 2023
VALID_YEAR = 2024
SEED = 42

# Negative:positive ratios to sweep. Values < 1 shrink the sample on purpose.
RATIOS = (0.25, 0.5, 1.0, 2.0)


def _years(df: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(df["acq_date"], errors="coerce").dt.year.to_numpy()


def _draw_negatives(
    background_year: pd.DataFrame, n_target: int, seed: int
) -> pd.DataFrame:
    """Deterministic subsample of one year's pool (all rows when short).

    Mirrors ``occurrence.sample_background`` but accepts ``n_target`` below
    the positive count (ratios < 1).
    """
    pool = background_year.sort_values(BACKGROUND_ID_COL, kind="stable").reset_index(
        drop=True
    )
    if n_target >= len(pool):
        return pool
    picked = pool.sample(n=int(n_target), random_state=seed)
    return picked.sort_values(BACKGROUND_ID_COL, kind="stable").reset_index(drop=True)


def _prepare_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot (fit on train) + median imputation + per-fold scaling."""
    X_train, dummies = categorical_to_dummies(train[feature_cols], categorical_cols)
    X_test, _ = categorical_to_dummies(
        test[feature_cols], categorical_cols, train_columns=dummies
    )
    cont = continuous_columns(X_train)
    medians = X_train[cont].median()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[cont] = X_train[cont].fillna(medians)
    X_test[cont] = X_test[cont].fillna(medians)
    scaler = fold_scaler_fit(X_train, cont)
    return (
        fold_scaler_apply(X_train, scaler, cont),
        fold_scaler_apply(X_test, scaler, cont),
    )


def _build_frame(
    positives: pd.DataFrame, background: pd.DataFrame, ratio: float
) -> tuple[pd.DataFrame, dict]:
    """Assemble the modeling frame at ``ratio`` negatives/positive per year."""
    pos_year = _years(positives)
    bg_year = _years(background)
    n_pos = {y: int((pos_year == y).sum()) for y in (TRAIN_YEAR, VALID_YEAR)}

    parts: list[pd.DataFrame] = []
    neg_year_parts: list[np.ndarray] = []
    for year in (TRAIN_YEAR, VALID_YEAR):
        pool = background.loc[bg_year == year]
        target = round(ratio * n_pos[year])
        drawn = _draw_negatives(pool, target, SEED)
        parts.append(drawn)
        neg_year_parts.append(np.full(len(drawn), year))

    negatives = pd.concat(parts, ignore_index=True)
    frame = assemble_modeling_frame(
        occurrence_features(positives), occurrence_features(negatives)
    )
    frame["year"] = np.concatenate(
        [pos_year, np.concatenate(neg_year_parts).astype(int)]
    )
    meta = {
        "n_pos_train": n_pos[TRAIN_YEAR],
        "n_pos_valid": n_pos[VALID_YEAR],
        "n_neg_train": int((frame["year"] == TRAIN_YEAR)[frame[LABEL_COL] == 0].sum()),
        "n_neg_valid": int((frame["year"] == VALID_YEAR)[frame[LABEL_COL] == 0].sum()),
    }
    return frame, meta


@pytest.mark.experiment
def test_temporal_ratio_experiment(capsys: pytest.CaptureFixture[str]) -> None:
    positives = load_csv(POSITIVES_CSV)
    background = load_background_days()
    report = validate_background_days(background, positives)
    assert report["n_duplicate_ids"] == 0
    assert report["n_leakage"] == 0

    rows: list[dict] = []
    for ratio in RATIOS:
        frame, meta = _build_frame(positives, background, ratio)
        train = frame[frame["year"] == TRAIN_YEAR]
        valid = frame[frame["year"] == VALID_YEAR]
        assert train[LABEL_COL].nunique() == 2
        assert valid[LABEL_COL].nunique() == 2

        X_train, X_valid = _prepare_fold(
            train, valid, EVERYDAY_FEATURES, categorical_columns()
        )
        y_train = train[LABEL_COL].to_numpy()
        y_valid = valid[LABEL_COL].to_numpy()
        prevalence = float(y_valid.mean())

        for name, model in model_candidates(seed=SEED).items():
            fitted = clone(model).fit(X_train, y_train)
            scores = fitted.predict_proba(X_valid)[:, 1]
            metrics = rank_metrics(y_valid, scores)
            rows.append(
                {
                    "ratio": ratio,
                    "model": name,
                    "n_train": len(train),
                    "train_neg_pos": meta["n_neg_train"] / meta["n_pos_train"],
                    "n_valid": len(valid),
                    "valid_neg_pos": meta["n_neg_valid"] / meta["n_pos_valid"],
                    "valid_prevalence": prevalence,
                    "pr_auc": metrics["pr_auc"],
                    "roc_auc": metrics["roc_auc"],
                }
            )

    table = pd.DataFrame(rows)
    assert np.isfinite(table[["pr_auc", "roc_auc"]].to_numpy()).all()
    assert table[["pr_auc", "roc_auc"]].to_numpy().min() >= 0.0
    assert table[["pr_auc", "roc_auc"]].to_numpy().max() <= 1.0

    order = {
        "logistic": 0,
        "logistic_balanced": 1,
        "random_forest": 2,
        "random_forest_balanced": 3,
        "hist_gradient_boosting": 4,
    }
    table = table.sort_values(
        ["ratio", "model"], key=lambda s: s.map(order) if s.name == "model" else s
    )
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print("\n== Per-ratio x model (train=2023, valid=2024) ==")
        print(table.to_string(index=False))
        print("\n== Mean over models, per ratio ==")
        print(
            table.groupby("ratio")[
                [
                    "train_neg_pos",
                    "valid_neg_pos",
                    "valid_prevalence",
                    "pr_auc",
                    "roc_auc",
                ]
            ]
            .mean()
            .to_string()
        )
        print("\n== Best model per ratio ==")
        print(
            table.loc[table.groupby("ratio")["pr_auc"].idxmax()][
                ["ratio", "model", "valid_prevalence", "pr_auc", "roc_auc"]
            ].to_string(index=False)
        )
