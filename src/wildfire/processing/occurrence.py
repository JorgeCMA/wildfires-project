"""Fire-occurrence modeling contract (fire / background ranking).

The occurrence model predicts *when/where fires are likely* from everyday
(non-satellite) conditions. Positives are FIRMS detections
(``wildfire_data_cleaned.csv``); negatives are random background CCAA x day
rows built later with the identical everyday schema. Output is a **risk
ranking** (PR-AUC), not a calibrated probability: prevalence is fixed by
the background sampling, not by nature.

Column rules enforced here:

- Satellite-measured columns can never enter ``X``: they are undefined
  for background rows (no brightness where nothing burned), so any model
  using them could not score a background day at all.
- Sub-daily columns (``acq_time`` and derivatives) are dropped: FIRMS only
  observes at satellite overpass times, so hour-of-day would be an
  "is-a-detection" artifact rather than a fire signal. Background rows
  carry date grain only; calendar features (month, day-of-year, weekend /
  holiday flags) stay.
- Derived meteo indices stay: they are functions of allowed inputs.

Deliberately excluded although non-satellite: ``indice_meteorologico`` /
``nivel_relativo`` / ``componentes_disponibles`` (deterministic functions
of the weather predictors — same leakage class as the confidence twin),
``daynight`` / ``is_day`` and every ``acq_time`` derivative (sub-daily
overpass artifacts, not fire signals at date grain), ``rain`` (identical
to ``precipitation``), ``acq_date`` / ``year`` (temporal signal carried by
month, day-of-year and season instead).

Settings live under ``modeling.occurrence`` in ``configs/project.yaml``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut  # type: ignore[import-untyped]

from wildfire.config import load_config
from wildfire.processing.pca_weather import WIND_DIRECTION_COL, AddWindCircular

LABEL_COL = "label"
SEASON_COL = "season"

# Everyday (non-satellite) predictors in fixed order: location, date-grain
# calendar, the 15 weather variables (precipitation, not rain; wd_sin/wd_cos
# instead of the raw circular direction), CLC center + 8 neighbors, region,
# budgets and derived meteo indices.
EVERYDAY_FEATURES: list[str] = [
    "latitude",
    "longitude",
    "month",
    "day_of_year",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "riesgo_antropico_alto",
    "temperature_2m",
    "precipitation",
    "surface_pressure",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wd_sin",
    "wd_cos",
    "wind_gusts_10m",
    "et0_fao_evapotranspiration",
    "shortwave_radiation",
    "boundary_layer_height",
    "vapour_pressure_deficit",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "clc_class",
    "clc_class_N",
    "clc_class_S",
    "clc_class_W",
    "clc_class_E",
    "clc_class_NW",
    "clc_class_NE",
    "clc_class_SW",
    "clc_class_SE",
    "clc_uniform_surroundings",
    "ccaa",
    "prevention_current",
    "prevention_prev3",
    "extinction_prev3",
    "ratio_prevencion_extincion",
    "euros_prevencion_por_ha",
    "euros_extincion_por_ha",
    "potencial_propagacion",
    "indice_riesgo_meteo",
    "indice_riesgo_meteo_log",
    "estres_termico_vegetacion",
]

# Satellite-measured columns: undefined for background rows, banned from X.
# Includes the confidence family (predicting it from everyday data would be
# sensor-characterization, not fire modeling) and rain/is_day duplicates.
SATELLITE_BLOCKLIST: list[str] = [
    "brightness",
    "brightness_ir",
    "scan",
    "frp",
    "type",
    "confidence",
    "confidence_num",
    "confidence_cat",
    "confidence_og_num",
    "confidence_og_cat",
    "rain",
    "is_day",
]

# Overpass-time artifacts: detections cluster at satellite overpass hours,
# so sub-daily features would mark "is-a-detection" instead of fire risk.
SUBDAILY_DROP: list[str] = [
    "acq_time",
    "acq_time_str",
    "hora",
    "franja_horaria",
    "daynight",
    "is_day",
]


def occurrence_settings() -> dict:
    """Settings from ``modeling.occurrence`` in config (validated)."""
    try:
        settings = dict(load_config()["modeling"]["occurrence"])
    except KeyError as exc:
        raise KeyError("Missing 'modeling.occurrence' in configs/project.yaml") from exc
    ratio = int(settings.get("background_ratio", 0))
    if ratio < 1:
        raise ValueError(f"background_ratio must be >= 1, got {ratio!r}")
    groups = list(settings.get("cv_groups", []))
    if not groups:
        raise ValueError("cv_groups must list at least one blocking variable")
    settings["background_ratio"] = ratio
    settings["cv_groups"] = groups
    return settings


def _with_wind_circular(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with ``wd_sin``/``wd_cos`` derived when absent."""
    if "wd_sin" in df.columns and "wd_cos" in df.columns:
        return df.copy()
    if WIND_DIRECTION_COL not in df.columns:
        raise KeyError("Missing wind direction (wd_sin/wd_cos or raw)")
    # Pass-through for every other column (label included); only the raw
    # circular direction is replaced by its sin/cos expansion.
    return AddWindCircular().fit_transform(df)


def occurrence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Everyday predictor matrix in fixed order (fail-closed).

    Missing allowlist columns raise ``KeyError``; any satellite or
    sub-daily column present is dropped with a warning (never silently).
    ``wd_sin``/``wd_cos`` are derived from ``wind_direction_10m`` when
    absent (same expansion as the weather PCA).
    """
    frame = _with_wind_circular(df)
    missing = [c for c in EVERYDAY_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing occurrence features: {missing}")
    banned = [c for c in SATELLITE_BLOCKLIST + SUBDAILY_DROP if c in df.columns]
    # EVERYDAY_FEATURES never contains banned columns, so selecting it
    # already excludes them; the warning keeps the exclusion audible.
    if banned:
        warnings.warn(
            f"Ignoring {len(banned)} satellite/sub-daily columns "
            f"({sorted(set(banned))}); they cannot enter the occurrence model"
        )
    return frame[EVERYDAY_FEATURES].copy()


def add_season(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``season`` (``YYYY-Qn``) from ``acq_date`` for blocking."""
    if "acq_date" not in df.columns:
        raise KeyError("Missing column: acq_date")
    out = df.copy()
    dates = pd.to_datetime(out["acq_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("acq_date has unparseable values")
    out[SEASON_COL] = dates.dt.year.astype(str) + "-Q" + dates.dt.quarter.astype(str)
    return out


def assemble_modeling_frame(
    positives: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame:
    """Join fire rows (label 1) and background rows (label 0).

    Both frames must carry the identical everyday schema; any satellite
    column in either frame raises ``ValueError``.
    """
    for name, frame in (("positives", positives), ("background", background)):
        banned = [c for c in SATELLITE_BLOCKLIST if c in frame.columns]
        if banned:
            raise ValueError(f"{name} carries satellite columns: {banned}")
    pos = _with_wind_circular(positives)
    neg = _with_wind_circular(background)
    missing = [c for c in EVERYDAY_FEATURES if c not in neg.columns]
    if missing:
        raise KeyError(f"background missing occurrence features: {missing}")
    pos = pos[EVERYDAY_FEATURES].copy()
    pos[LABEL_COL] = 1
    neg = neg[EVERYDAY_FEATURES].copy()
    neg[LABEL_COL] = 0
    frame = pd.concat([pos, neg], ignore_index=True)
    if frame[LABEL_COL].nunique() != 2:
        raise ValueError("Modeling frame needs both labels (empty input?)")
    return frame


def background_skeleton(
    ccaa: list[str],
    start: str,
    end: str,
    n_per_day: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Candidate CCAA x day grid for later background sampling.

    Pure skeleton (region + date only): weather/CLC are filled afterwards
    with the existing enrichment machinery. Deterministic under ``seed``.
    """
    if not ccaa:
        raise ValueError("ccaa must list at least one region")
    if n_per_day < 1:
        raise ValueError(f"n_per_day must be >= 1, got {n_per_day!r}")
    days = pd.date_range(start, end, freq="D")
    if len(days) == 0:
        raise ValueError(f"Empty date range: {start}..{end}")
    rng = np.random.default_rng(seed)
    picks = rng.choice(ccaa, size=(len(days), n_per_day))
    return pd.DataFrame(
        {
            "ccaa": picks.ravel(),
            "acq_date": np.repeat(days.date.astype(str), n_per_day),
        }
    )


@dataclass
class BlockedSplit:
    """One leave-one-group-out split (train/test row positions)."""

    group: str
    held_out: str
    train_index: np.ndarray
    test_index: np.ndarray


def blocked_splits(df: pd.DataFrame, group: str) -> list[BlockedSplit]:
    """Leave-one-group-out splits over ``group`` (``season`` or ``ccaa``).

    Random K-fold would leak via nearby days/places and overstate ranking
    quality; blocking keeps each held-out group strictly out of training.
    """
    if group == SEASON_COL and SEASON_COL not in df.columns:
        raise KeyError("Missing 'season': run add_season() first")
    if group not in df.columns:
        raise KeyError(f"Missing blocking column: {group}")
    groups = df[group].to_numpy()
    splits = []
    for train_pos, test_pos in LeaveOneGroupOut().split(df, groups=groups):
        splits.append(
            BlockedSplit(
                group=group,
                held_out=str(groups[test_pos[0]]),
                train_index=np.asarray(train_pos),
                test_index=np.asarray(test_pos),
            )
        )
    if not splits:
        raise ValueError(f"No groups to split on in column: {group}")
    return splits


def rank_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray) -> dict:
    """Ranking quality of fire-likelihood scores (higher is better)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("rank_metrics needs both labels present")
    if not np.isfinite(s).all():
        raise ValueError("rank_metrics needs finite scores")
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }
