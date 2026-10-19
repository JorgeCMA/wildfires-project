"""Persisted occurrence-model bundle for the live Plan-C point scorer.

The notebook owns the experiment loop; this module captures the *deployment*
slice of it — everything a live request needs to reproduce the notebook's
preprocessing, score a point and turn that score into a calibrated exceedance
probability:

- :class:`Preprocessing` — one-hot columns, median imputations, per-column
  scaler and the exact training column order (mirrors ``prepare_fold`` but fit
  once and reused at inference).
- :class:`MarginCalibrator` — the per-margin isotonic calibrator plus the
  source/target prevalences needed by ``prior_correct``.
- :class:`OccurrenceBundle` — the fitted estimator wired to both, with
  :func:`save_bundle`/:func:`load_bundle` (joblib) and the inference helpers
  :func:`predict_score` / :func:`calibrated_exceedance` / :func:`score_point`.

The bundle is produced by ``scripts/fit_occurrence_model.py``; nothing here
touches the network or the raw data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from wildfire.analysis.calibration import apply_isotonic, prior_correct
from wildfire.analysis.occurrence_model import (
    categorical_columns,
    categorical_to_dummies,
    continuous_columns,
    fold_scaler_apply,
    fold_scaler_fit,
)
from wildfire.config import PROJECT_ROOT
from wildfire.processing.occurrence import EVERYDAY_FEATURES


@dataclass
class Preprocessing:
    """Fold-fit preprocessing replayed at inference (see ``prepare_fold``).

    Training order is ``feature_cols`` minus the nominal columns (as one-hot
    dummies) — stored in :attr:`feature_order` so a live request is reindexed to
    the exact matrix the estimator saw.
    """

    feature_cols: list[str]
    categorical: list[str]
    dummies: list[str]
    continuous: list[str]
    medians: dict[str, float]
    scaler: StandardScaler
    feature_order: list[str] = field(default_factory=list)


@dataclass
class MarginCalibrator:
    """Isotonic calibrator + prevalences for one exceedance margin."""

    margin_km: float
    calibrator: Any
    source_prevalence: float
    target_prevalence: float


@dataclass
class OccurrenceBundle:
    """Fitted estimator + preprocessing + per-margin calibration + metadata."""

    model: Any
    preprocessing: Preprocessing
    margins: dict[float, MarginCalibrator]
    metadata: dict = field(default_factory=dict)


def fit_preprocessing(
    train: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    categorical_cols: Sequence[str] | None = None,
) -> Preprocessing:
    """Fit the deployment preprocessing on a training frame (mirrors ``prepare_fold``).

    One-hot is fit on ``train``; the median imputation and the scaler are fit on
    the encoded continuous columns, exactly as the notebook's per-fold
    ``prepare_fold`` does for each train fold.
    """
    feature_cols = list(feature_cols or EVERYDAY_FEATURES)
    categorical_cols = list(categorical_cols or categorical_columns())

    encoded, dummies = categorical_to_dummies(train[feature_cols], categorical_cols)
    cont = continuous_columns(encoded)
    medians = encoded[cont].median()
    encoded[cont] = encoded[cont].fillna(medians)
    scaler = fold_scaler_fit(encoded, cont)

    return Preprocessing(
        feature_cols=feature_cols,
        categorical=categorical_cols,
        dummies=list(dummies),
        continuous=list(cont),
        medians={c: float(v) for c, v in medians.items()},
        scaler=scaler,
        feature_order=list(encoded.columns),
    )


def transform(frame: pd.DataFrame, preprocessing: Preprocessing) -> pd.DataFrame:
    """Encode, impute, scale and reorder a frame to the training matrix layout.

    Unseen categorical levels become all-zero dummies; continuous columns absent
    from ``frame`` are filled with the training medians; the result is reindexed
    to :attr:`Preprocessing.feature_order`.
    """
    encoded, _ = categorical_to_dummies(
        frame.reindex(columns=preprocessing.feature_cols),
        preprocessing.categorical,
        train_columns=preprocessing.dummies,
    )
    encoded = encoded.copy()
    for column in preprocessing.continuous:
        if column not in encoded.columns:
            encoded[column] = np.nan
    encoded[preprocessing.continuous] = encoded[preprocessing.continuous].fillna(
        preprocessing.medians
    )
    scaled = fold_scaler_apply(encoded, preprocessing.scaler, preprocessing.continuous)
    return scaled.reindex(columns=preprocessing.feature_order)


def predict_score(bundle: OccurrenceBundle, features: pd.DataFrame) -> np.ndarray:
    """Raw (uncalibrated) occurrence score for each row of ``features``."""
    matrix = transform(features, bundle.preprocessing)
    scores = bundle.model.predict_proba(matrix)[:, 1]
    return np.asarray(scores, dtype=float)


def calibrated_exceedance(
    bundle: OccurrenceBundle, score: float | np.ndarray
) -> dict[float, float]:
    """Calibrated exceedance probability per margin from a raw score.

    Applies each margin's isotonic map then the odds correction from the
    calibration-frame prevalence to the measured base rate.
    """
    values = np.atleast_1d(np.asarray(score, dtype=float))
    out: dict[float, float] = {}
    for margin, entry in bundle.margins.items():
        calibrated = apply_isotonic(entry.calibrator, values)
        corrected = prior_correct(
            calibrated,
            source_prevalence=entry.source_prevalence,
            target_prevalence=entry.target_prevalence,
        )
        out[float(margin)] = float(np.asarray(corrected).reshape(-1)[0])
    return out


def score_point(bundle: OccurrenceBundle, features: pd.DataFrame) -> dict:
    """One-row score: raw value, per-margin exceedance and the default band.

    ``features`` is the single-row frame from
    :func:`wildfire.analysis.inference.build_point_features`.
    """
    scores = predict_score(bundle, features)
    score = float(scores[0])
    return {
        "score": score,
        "exceedance": calibrated_exceedance(bundle, score),
    }


def _resolve(path: str | Path) -> Path:
    """Anchor a relative bundle path to the project root (absolute kept as-is)."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def save_bundle(bundle: OccurrenceBundle, path: str | Path) -> Path:
    """Serialize ``bundle`` with joblib, creating parent directories."""
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> OccurrenceBundle:
    """Load a bundle written by :func:`save_bundle` (fails if absent)."""
    path = _resolve(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Occurrence bundle not found at {path}: run scripts/fit_occurrence_model.py"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, OccurrenceBundle):
        raise TypeError(f"{path} does not contain an OccurrenceBundle")
    return bundle


__all__ = [
    "MarginCalibrator",
    "OccurrenceBundle",
    "Preprocessing",
    "calibrated_exceedance",
    "fit_preprocessing",
    "load_bundle",
    "predict_score",
    "save_bundle",
    "score_point",
    "transform",
]
