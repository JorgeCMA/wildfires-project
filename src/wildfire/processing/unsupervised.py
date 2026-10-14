"""Unsupervised-learning feature sets (block 03).

Single source of truth for the ENTRADA_*/PERFILADO column lists from
``planning/plan_unsupervised_implementation.md`` §5. Notebook 03a displays
and orchestrates; the units that can break silently live here, tested.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from wildfire.config import load_config
from wildfire.processing.occurrence import SATELLITE_BLOCKLIST
from wildfire.processing.pca_weather import AddWindCircular, pca_feature_columns

#: Fallback when ``unsupervised.min_cliff_delta`` is absent from config.
DEFAULT_MIN_CLIFF_DELTA = 0.2

#: Right-skewed columns needing ``log1p`` before scaling (plan §5).
LOG_COLS: list[str] = [
    "precipitation",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
]

#: Seasonality pairs, already in [-1, 1]: never scaled (plan §5, set B).
SEASON_SINCOS: list[str] = [
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

#: Sole representative of time of day (plan §5, set B).
TIME_COL = "franja_horaria"

#: The 8 CLC neighbour columns; directions collapse (plan §5, set C).
CLC_NEIGHBOURS: list[str] = [
    "clc_class_N",
    "clc_class_S",
    "clc_class_W",
    "clc_class_E",
    "clc_class_NW",
    "clc_class_NE",
    "clc_class_SW",
    "clc_class_SE",
]

#: Prefix for per-class neighbour-frequency columns built by
#: :func:`build_clc_environment`.
FREQ_VECINA_PREFIX = "freq_vecina_"

#: Fixed CLC-environment columns (frequencies are data-dependent, see
#: :func:`clc_environment_columns`).
CLC_ENV_FIXED: list[str] = [
    "pct_vecinos_igual_centro",
    "n_clases_distintas_vecinos",
]

#: Set A: the 14 weather-PCA features (12 base + wd pair).
ENTRADA_A: list[str] = pca_feature_columns()

#: Set B: A + seasonality + time of day.
ENTRADA_B: list[str] = [*ENTRADA_A, *SEASON_SINCOS, TIME_COL]

#: Set C (fixed part): B + centre pixel + collapsed neighbourhood.
#: Append :func:`clc_environment_columns` output for the frequency columns.
ENTRADA_C_FIXED: list[str] = [*ENTRADA_B, "clc_class", *CLC_ENV_FIXED]

#: Set D: pure spatial (hotspots, plan §5-D). Project to km in the notebook.
ENTRADA_D: list[str] = ["latitude", "longitude", "acq_date"]

#: Profiling / post-hoc validation only: never algorithm inputs (plan §5).
PERFILADO: list[str] = [
    "frp",
    "brightness",
    "brightness_ir",
    "scan",
    "confidence_num",
    "confidence_cat",
    "ccaa",
    "year",
    "month",
    "day_of_year",
    "daynight",
    "is_day",
    "acq_time",
    "acq_date",
    "type",
    "rain",
    "wind_direction_10m",
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
    "indice_meteorologico",
    "riesgo_antropico_alto",
    "nivel_relativo",
    "componentes_disponibles",
]


def load_min_cliff_delta() -> float:
    """Read ``unsupervised.min_cliff_delta`` (plan §7B).

    Falls back to :data:`DEFAULT_MIN_CLIFF_DELTA` when the key is missing
    so older configs keep working.
    """
    try:
        value = load_config().get("unsupervised", {}).get(
            "min_cliff_delta", DEFAULT_MIN_CLIFF_DELTA
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):  # fail-closed
        warnings.warn(
            "Could not read unsupervised.min_cliff_delta; "
            f"using default {DEFAULT_MIN_CLIFF_DELTA}.",
            UserWarning,
            stacklevel=2,
        )
        return DEFAULT_MIN_CLIFF_DELTA
    return float(value)


def add_wind_circular(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``wd_sin``/``wd_cos`` derived from ``wind_direction_10m``.

    Thin wrapper over :class:`AddWindCircular`; the raw circular column
    stays out of every feature set (plan §4, rule 2).
    """
    out = df.copy()
    derived = AddWindCircular().fit_transform(out[["wind_direction_10m"]])
    out[["wd_sin", "wd_cos"]] = derived
    return out


def apply_log_transforms(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Apply ``log1p`` to right-skewed columns (plan §5).

    Missing columns warn and are skipped (fail-closed, repo style).
    """
    out = df.copy()
    for col in LOG_COLS if columns is None else columns:
        if col not in out.columns:
            warnings.warn(
                f"log1p skipped: {col!r} not in frame.", UserWarning, stacklevel=2
            )
            continue
        out[col] = np.log1p(out[col].clip(lower=0))
    return out


def build_clc_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 8 CLC neighbour directions into environment features.

    Returns ``pct_vecinos_igual_centro`` (share of neighbours matching the
    centre pixel), ``n_clases_distintas_vecinos`` (1–8) and one
    ``freq_vecina_<clase>`` column per observed class. Missing neighbour
    columns warn and yield NaN (fail-closed).
    """
    missing = [c for c in ["clc_class", *CLC_NEIGHBOURS] if c not in df.columns]
    env = pd.DataFrame(index=df.index)
    if missing:
        warnings.warn(
            f"CLC environment incomplete, missing: {missing}.",
            UserWarning,
            stacklevel=2,
        )
        env["pct_vecinos_igual_centro"] = np.nan
        env["n_clases_distintas_vecinos"] = np.nan
        return env
    neigh = df[CLC_NEIGHBOURS]
    centre = df["clc_class"]
    env["pct_vecinos_igual_centro"] = neigh.eq(centre, axis=0).mean(axis=1)
    env["n_clases_distintas_vecinos"] = neigh.nunique(axis=1)
    for cls in sorted(neigh.stack().dropna().unique().tolist()):
        env[f"{FREQ_VECINA_PREFIX}{cls}"] = (neigh == cls).mean(axis=1)
    return env


def clc_environment_columns(env: pd.DataFrame) -> list[str]:
    """List the derived CLC-environment columns of a built frame."""
    return [
        c
        for c in env.columns
        if c in CLC_ENV_FIXED or c.startswith(FREQ_VECINA_PREFIX)
    ]


def assert_feature_sets_clean(*sets: list[str]) -> None:
    """Fail closed when a satellite column leaks into an input set.

    Satellite readings describe the sensor, not the cause (plan §4, rule 1):
    their presence in inputs is a bug, so this raises instead of warning.
    """
    for columns in sets:
        leaked = [c for c in columns if c in SATELLITE_BLOCKLIST]
        if leaked:
            raise ValueError(
                f"Satellite columns in unsupervised inputs: {leaked}. "
                "They belong to PERFILADO, never to ENTRADA_*."
            )


def resolve_set(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    """Select an input set, warning (not raising) on absent columns."""
    absent = [c for c in columns if c not in df.columns]
    if absent:
        warnings.warn(
            f"{name}: absent columns dropped: {absent}.", UserWarning, stacklevel=2
        )
    return df[[c for c in columns if c in df.columns]].copy()
