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
from pathlib import Path

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

#: Key of the enriched background file (PLAN §A file names). Duplicated from
#: ``reintegration.BACKGROUND_ID_COL`` on purpose: importing it here would
#: close an import cycle (``occurrence`` <- ``unsupervised`` <- ``reintegration``).
BACKGROUND_ID_COL = "background_id"

#: Mean Earth radius (km) for the haversine leakage radius.
_EARTH_RADIUS_KM = 6371.0088

#: Satellital columns that may legitimately ride along in the background file
#: because they are *weather* outputs, not satellite measurements:
#: ``occurrence_features``/``assemble_modeling_frame`` drop ``rain``/``is_day``
#: downstream, and ``type`` is structural (allowed only when all-zero).
_BACKGROUND_ALLOWED_LEFTOVERS = frozenset({"rain", "is_day", "type"})

#: Reintegration candidates (PLAN §B.2.4): produced for both classes by
#: fitted-on-positives objects (``processing/reintegration.py``). They enter
#: the modeling frame ONLY via the ``candidates`` argument of
#: ``assemble_modeling_frame`` and are promoted into ``EVERYDAY_FEATURES``
#: only if the ablation gate passes (mean ΔPR-AUC > 0, PLAN §C.2). Until then
#: they "exist but are unproven" in the methods note.
CANDIDATE_FEATURES: list[str] = ["cluster_meteo", "score_anomalia"]

#: Posterior columns of the elected k12 GMM: the alternative encoding of the
#: ``cluster_meteo`` candidate (12 continuous levels, PLAN §B.2.2). Kept as a
#: literal here on purpose: ``unsupervised`` imports this module, so importing
#: ``typology`` (which imports ``unsupervised``) would cycle.
_K12_POSTERIOR_K = 12
CANDIDATE_POSTERIOR_FEATURES: list[str] = [
    f"gmm_k12_full_p{j}" for j in range(_K12_POSTERIOR_K)
]

#: Everything a candidate block may carry (hard label OR its posterior form).
_CANDIDATE_ALLOWED: list[str] = [*CANDIDATE_FEATURES, *CANDIDATE_POSTERIOR_FEATURES]

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
    leakage = float(settings.get("leakage_max_km", 1.0))
    if leakage < 0:
        raise ValueError(f"leakage_max_km must be >= 0, got {leakage!r}")
    risk_bands = int(settings.get("risk_bands", 5))
    if risk_bands < 2:
        raise ValueError(f"risk_bands must be >= 2, got {risk_bands!r}")
    margins = [
        float(m) for m in settings.get("calibration_margins_km", [5.0, 10.0, 20.0])
    ]
    if not margins or any(m <= 0 for m in margins):
        raise ValueError(f"calibration_margins_km must be positive, got {margins!r}")
    settings["background_ratio"] = ratio
    settings["cv_groups"] = groups
    settings["leakage_max_km"] = leakage
    settings["risk_bands"] = risk_bands
    settings["calibration_margins_km"] = margins
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


def _candidate_block(
    pos_candidates: pd.DataFrame | None,
    neg_candidates: pd.DataFrame | None,
    n_pos: int,
    n_neg: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate the optional reintegration-candidate frames (PLAN §B.2.4).

    Both frames or neither; identical canonical column order (a subset of
    ``CANDIDATE_FEATURES`` or its posterior alternative
    ``CANDIDATE_POSTERIOR_FEATURES``, reindexed to the fixed order); row
    counts match the positives/background frames; no satellite columns.
    """
    if (pos_candidates is None) != (neg_candidates is None):
        raise ValueError("pos_candidates and neg_candidates must come together")
    if pos_candidates is None or neg_candidates is None:
        return pd.DataFrame(index=range(n_pos)), pd.DataFrame(index=range(n_neg)), []
    cols = list(dict.fromkeys(pos_candidates.columns))
    ordered = [c for c in _CANDIDATE_ALLOWED if c in cols]
    for name, frame, expected in (
        ("pos_candidates", pos_candidates, n_pos),
        ("neg_candidates", neg_candidates, n_neg),
    ):
        if len(frame) != expected:
            raise ValueError(f"{name} has {len(frame)} rows, expected {expected}")
        banned = [c for c in SATELLITE_BLOCKLIST if c in frame.columns]
        if banned:
            raise ValueError(f"{name} carries satellite columns: {banned}")
    unknown = [c for c in cols if c not in _CANDIDATE_ALLOWED]
    if unknown:
        raise ValueError(f"Candidate columns outside the candidate set: {unknown}")
    pos = pos_candidates[ordered].reset_index(drop=True)
    neg = neg_candidates[ordered].reset_index(drop=True)
    return pos, neg, ordered


def assemble_modeling_frame(
    positives: pd.DataFrame,
    background: pd.DataFrame,
    *,
    pos_candidates: pd.DataFrame | None = None,
    neg_candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join fire rows (label 1) and background rows (label 0).

    Both frames must carry the identical everyday schema; any satellite
    column in either frame raises ``ValueError``. Reintegration candidates
    (``cluster_meteo``, ``score_anomalia``, or the posterior encoding) are
    appended after the everyday block in canonical order when both candidate
    frames are given; they are optional and never mixed into
    ``EVERYDAY_FEATURES``.
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
    pos_c, neg_c, candidate_cols = _candidate_block(
        pos_candidates, neg_candidates, len(pos), len(neg)
    )
    pos = pos[EVERYDAY_FEATURES].copy().reset_index(drop=True)
    neg = neg[EVERYDAY_FEATURES].copy().reset_index(drop=True)
    if candidate_cols:
        pos = pd.concat([pos, pos_c], axis=1)
        neg = pd.concat([neg, neg_c], axis=1)
    pos[LABEL_COL] = 1
    neg[LABEL_COL] = 0
    frame = pd.concat([pos, neg], ignore_index=True)
    if frame[LABEL_COL].nunique() != 2:
        raise ValueError("Modeling frame needs both labels (empty input?)")
    return frame


def load_background_days(path: str | Path | None = None) -> pd.DataFrame:
    """Load the enriched background file (key ``background_id``).

    ``path=None`` reads ``output.background_days`` from
    ``configs/project.yaml``; an explicit path overrides it (tests, reruns).
    Fails closed if the key column is absent.
    """
    # Import diferido: `wildfire.data` carga requests-cache y el cliente
    # Open-Meteo; el módulo de ocurrencia no debe pagar ese coste al importar.
    from wildfire.data.io import load_csv

    if path is None:
        try:
            path = load_config()["output"]["background_days"]
        except KeyError as exc:
            raise KeyError(
                "Missing 'output.background_days' in configs/project.yaml"
            ) from exc
    background = load_csv(path)
    if BACKGROUND_ID_COL not in background.columns:
        raise KeyError(
            f"Background file {path!r} lacks key column {BACKGROUND_ID_COL!r}"
        )
    return background


def _same_day_collisions(
    background: pd.DataFrame,
    positives: pd.DataFrame,
    leakage_max_km: float,
) -> np.ndarray:
    """Boolean mask of background rows within ``leakage_max_km`` of a same-day
    positive detection (haversine, date-partitioned spatial index).
    """
    from sklearn.neighbors import BallTree  # type: ignore[import-untyped]

    required = ("latitude", "longitude", "acq_date")
    for name, frame in (("background", background), ("positives", positives)):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} missing columns for the leakage check: {missing}")

    pos = positives[list(required)].dropna(subset=["latitude", "longitude"])
    dates = pd.to_datetime(pos["acq_date"], errors="coerce")
    keep = dates.notna().to_numpy()
    coords = np.radians(pos[["latitude", "longitude"]].to_numpy(dtype=float))[keep]
    day = dates.dt.strftime("%Y-%m-%d").to_numpy()[keep]

    by_day: dict[str, np.ndarray] = {}
    for d in np.unique(day):
        by_day[d] = coords[day == d]

    bg = background[list(required)]
    bg_dates = pd.to_datetime(bg["acq_date"], errors="coerce")
    bg_coords = np.radians(bg[["latitude", "longitude"]].to_numpy(dtype=float))
    bg_day = bg_dates.dt.strftime("%Y-%m-%d").to_numpy()
    # Rows with NaN coordinates/dates are reported by the NaN report and
    # skipped here instead of crashing the BallTree query.
    finite = np.isfinite(bg_coords).all(axis=1)

    radius = float(leakage_max_km) / _EARTH_RADIUS_KM
    hits = np.zeros(len(background), dtype=bool)
    for d in np.unique(bg_day):
        day_pos = by_day.get(d)
        if day_pos is None:
            continue
        rows = np.flatnonzero((bg_day == d) & finite)
        if rows.size == 0:
            continue
        tree = BallTree(day_pos, metric="haversine")
        distance, _ = tree.query(bg_coords[rows], k=1)
        hits[rows] = distance[:, 0] <= radius
    return hits


def validate_background_days(
    background: pd.DataFrame,
    positives: pd.DataFrame,
    leakage_max_km: float = 1.0,
    *,
    allow_leakage: bool = False,
) -> dict:
    """Fail-closed contract checks on the enriched background file.

    Verifies the key is unique, the everyday schema is present, no
    satellite-measured column slipped in (``type`` is allowed only at 0;
    ``rain``/``is_day`` are weather outputs dropped downstream), and — unless
    ``allow_leakage`` — that no background row sits within ``leakage_max_km``
    of a same-day detection.

    Returns
    -------
    dict
        ``n_rows``, ``n_duplicate_ids``, ``n_leakage`` and the per-column
        ``nan_report`` (only columns with at least one NaN).
    """
    if BACKGROUND_ID_COL not in background.columns:
        raise KeyError(f"Missing key column: {BACKGROUND_ID_COL}")
    if "acq_date" not in background.columns:
        raise KeyError("Missing column: acq_date")

    frame = _with_wind_circular(background)
    missing = [c for c in EVERYDAY_FEATURES if c not in frame.columns]
    if missing:
        raise KeyError(f"Background missing occurrence features: {missing}")

    if "type" in background.columns:
        nonzero = int(
            (pd.to_numeric(background["type"], errors="coerce").fillna(0) != 0).sum()
        )
        if nonzero:
            raise ValueError(f"Background has {nonzero} rows with type != 0")
    banned = [
        c
        for c in SATELLITE_BLOCKLIST
        if c in background.columns and c not in _BACKGROUND_ALLOWED_LEFTOVERS
    ]
    if banned:
        raise ValueError(f"Background carries satellite columns: {banned}")

    n_duplicates = int(background[BACKGROUND_ID_COL].duplicated().sum())
    if n_duplicates:
        raise ValueError(
            f"background_id is not unique: {n_duplicates} duplicate row(s)"
        )

    nan_report = {
        c: int(background[c].isna().sum())
        for c in background.columns
        if background[c].isna().any()
    }

    hits = _same_day_collisions(background, positives, leakage_max_km)
    n_leakage = int(hits.sum())
    if n_leakage and not allow_leakage:
        raise ValueError(
            f"{n_leakage} background row(s) within {leakage_max_km} km of a "
            "same-day detection (label leakage); drop them or pass "
            "allow_leakage=True"
        )

    return {
        "n_rows": len(background),
        "n_duplicate_ids": n_duplicates,
        "n_leakage": n_leakage,
        "nan_report": nan_report,
    }


def drop_leaking_background(
    background: pd.DataFrame,
    positives: pd.DataFrame,
    leakage_max_km: float = 1.0,
) -> pd.DataFrame:
    """Return ``background`` without same-day leakage rows.

    Drops every row within ``leakage_max_km`` of a same-day detection — the
    condition :func:`validate_background_days` rejects on — so the saved
    artifact passes the contract as-is. The index is reset (positional).
    """
    hits = _same_day_collisions(background, positives, leakage_max_km)
    return background.loc[~hits].reset_index(drop=True)


def sample_background(
    background: pd.DataFrame,
    n_positives: int,
    *,
    ratio: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Seeded background subsample at ``ratio`` negatives per positive.

    Rows are sorted by ``background_id`` first (reproducible under any input
    order); when the pool is smaller than ``ratio * n_positives`` the whole
    pool is returned. ``ratio=None`` reads ``background_ratio`` from config.
    """
    if n_positives < 0:
        raise ValueError(f"n_positives must be >= 0, got {n_positives!r}")
    if ratio is None:
        ratio = int(occurrence_settings()["background_ratio"])
    if ratio < 1:
        raise ValueError(f"ratio must be >= 1, got {ratio!r}")
    pool = background.sort_values(BACKGROUND_ID_COL, kind="stable").reset_index(
        drop=True
    )
    target = int(ratio) * int(n_positives)
    if target >= len(pool):
        return pool
    picked = pool.sample(n=target, random_state=seed)
    return picked.sort_values(BACKGROUND_ID_COL, kind="stable").reset_index(drop=True)


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
