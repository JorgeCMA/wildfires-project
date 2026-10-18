"""Model units for the supervised occurrence model (Plan C.1).

The notebook owns the loop (blocked splits, per-fold fit, scoring, collecting
the metrics frame); this module only supplies the small, tested pieces those
cells call:

- :func:`model_candidates` — the provisional three estimators, fresh per call.
- :func:`fold_scaler_fit` / :func:`fold_scaler_apply` — make the per-fold
  scaler fit one visible call (hard rule 4.3), never hidden in a loop body.
- :func:`categorical_to_dummies` — one-hot encoding **fit on the train fold**
  and aligned across folds (unseen test categories become all-zero), so the
  encoding cannot leak.

Column policy is derived from ``EVERYDAY_FEATURES``: the nominal columns
(``ccaa``, ``clc_class``) are one-hot, everything else is continuous and
scaled. Reintegration candidates follow the same split (``cluster_meteo``
categorical, ``score_anomalia`` continuous) when present.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from wildfire.processing.occurrence import (
    CANDIDATE_FEATURES,
    CANDIDATE_POSTERIOR_FEATURES,
    EVERYDAY_FEATURES,
)

#: Nominal everyday columns: one-hot encoded, never standardized.
DEFAULT_CATEGORICAL: list[str] = ["ccaa", "clc_class"]

#: Reintegration candidates by kind (subset of ``CANDIDATE_FEATURES``). The k12
#: posterior columns are the continuous alternative to ``cluster_meteo``.
CANDIDATE_CATEGORICAL: list[str] = ["cluster_meteo"]
CANDIDATE_CONTINUOUS: list[str] = [
    c for c in CANDIDATE_FEATURES if c not in CANDIDATE_CATEGORICAL
] + list(CANDIDATE_POSTERIOR_FEATURES)

#: Fixed column policy derived from ``EVERYDAY_FEATURES``.
CATEGORICAL_FEATURES: list[str] = list(DEFAULT_CATEGORICAL)
CONTINUOUS_FEATURES: list[str] = [
    c for c in EVERYDAY_FEATURES if c not in CATEGORICAL_FEATURES
]


def categorical_columns(include_candidates: bool = False) -> list[str]:
    """Nominal columns to one-hot; candidate columns added on request."""
    cols = list(DEFAULT_CATEGORICAL)
    if include_candidates:
        cols.extend(c for c in CANDIDATE_CATEGORICAL if c not in cols)
    return cols


def continuous_columns(frame: pd.DataFrame) -> list[str]:
    """Continuous columns present in ``frame`` (everyday + candidates)."""
    everyday = [c for c in CONTINUOUS_FEATURES if c in frame.columns]
    candidates = [c for c in CANDIDATE_CONTINUOUS if c in frame.columns]
    return everyday + candidates


def model_candidates(seed: int = 42) -> dict[str, Any]:
    """The provisional model set, a fresh instance per call.

    ``LogisticRegression`` is the interpretable floor (permutation
    importance); each family also has a class-balanced variant because the
    background sampling is only ~1:2 (PLAN §C.1). ``RandomForestClassifier``
    and ``HistGradientBoostingClassifier`` are the non-linear candidates.
    sklearn only, sensible defaults; no tuning budget (optuna is OPEN).

    Class weights are a mild change at ~1:2, so the balanced and unbalanced
    variants are compared side by side rather than chosen a priori.
    """
    return {
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        "logistic_balanced": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=seed, n_jobs=-1
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def _dummy_name(column: str, value: Any) -> str:
    return f"{column}_{value}"


def categorical_to_dummies(
    frame: pd.DataFrame,
    cols: Sequence[str],
    train_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode ``cols``; fit on the train fold, align across folds.

    Parameters
    ----------
    frame:
        Modeling frame (continuous columns untouched).
    cols:
        Nominal columns to encode. Columns absent from ``frame`` are skipped
        (their dummies stay all-zero when applying a train encoding).
    train_columns:
        ``None`` fits on ``frame`` (call this on the train fold) and returns
        the produced dummy columns. Passing a previously returned list aligns
        ``frame`` to that encoding: unseen categories become all-zero.

    Returns
    -------
    (pd.DataFrame, list[str])
        The encoded frame (original nominal columns removed, dummies appended)
        and the canonical dummy-column list.
    """
    present = [c for c in cols if c in frame.columns]
    out = frame.drop(columns=present).copy()

    if train_columns is None:
        train_columns = []
        for col in present:
            values = sorted(pd.unique(frame[col].dropna()), key=str)
            train_columns.extend(_dummy_name(col, v) for v in values)

    dummies = pd.DataFrame(0.0, index=frame.index, columns=list(train_columns))
    for col in present:
        values = frame[col]
        for value in pd.unique(values.dropna()):
            name = _dummy_name(col, value)
            if name in dummies.columns:
                dummies.loc[values == value, name] = 1.0
    encoded = pd.concat([out, dummies], axis=1)
    return encoded, list(train_columns)


def fold_scaler_fit(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> StandardScaler:
    """Fit a ``StandardScaler`` on the continuous columns of the train fold."""
    cols = list(columns) if columns is not None else continuous_columns(frame)
    if not cols:
        raise ValueError("No continuous columns to scale")
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing continuous columns: {missing}")
    scaler = StandardScaler()
    scaler.fit(frame[cols].to_numpy(dtype=float))
    return scaler


def fold_scaler_apply(
    frame: pd.DataFrame, scaler: StandardScaler, columns: Sequence[str]
) -> pd.DataFrame:
    """Standardize ``columns`` with a scaler fit on the train fold."""
    cols = list(columns)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing continuous columns: {missing}")
    out = frame.copy()
    out[cols] = scaler.transform(frame[cols].to_numpy(dtype=float))
    return out
