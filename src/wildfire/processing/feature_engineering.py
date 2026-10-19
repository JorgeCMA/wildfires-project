"""Engineered features shared by the batch pipeline (notebook 01b) and inference.

Pure, deterministic transforms that turn an *enriched* frame (weather + CLC +
budgets already attached) into the model's engineered predictors: calendar,
holidays/antrópico flag, cyclic encoding, MITECO hectares, budget ratios and the
meteo indices. Ported from notebook 01b so the live point scorer
(``analysis/inference.py``) reproduces exactly the batch features the model was
trained on.
"""

from __future__ import annotations

from functools import cache

import holidays  # type: ignore[import-untyped]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from wildfire.config import load_config

#: MITECO forest area (ha) per CCAA (notebook 01b, cell 2.10).
SUP_FORESTAL_HA: dict[str, int] = {
    "Andalucía": 4_400_000,
    "Aragón": 2_600_000,
    "Asturias": 770_000,
    "Islas Baleares": 220_000,
    "Canarias": 570_000,
    "Cantabria": 360_000,
    "Castilla y León": 5_100_000,
    "Castilla-La Mancha": 3_600_000,
    "Cataluña": 2_000_000,
    "Comunidad Valenciana": 1_300_000,
    "Extremadura": 2_700_000,
    "Galicia": 2_000_000,
    "Madrid": 430_000,
    "Región de Murcia": 510_000,
    "Navarra": 610_000,
    "País Vasco": 490_000,
    "La Rioja": 340_000,
}

#: Budget-name CCAA → ``holidays.Spain`` subdivision code (notebook 01b, cell 2.8).
CCAA_HOLIDAY_CODES: dict[str, str] = {
    "Andalucía": "AN",
    "Aragón": "AR",
    "Asturias": "AS",
    "Islas Baleares": "IB",
    "Canarias": "CN",
    "Cantabria": "CB",
    "Castilla y León": "CL",
    "Castilla-La Mancha": "CM",
    "Cataluña": "CT",
    "Comunidad Valenciana": "VC",
    "Extremadura": "EX",
    "Galicia": "GA",
    "Madrid": "MD",
    "Región de Murcia": "MC",
    "Navarra": "NC",
    "País Vasco": "PV",
    "La Rioja": "RI",
}


@cache
def _holiday_calendar(year: int, code: str):
    return holidays.Spain(years=year, subdiv=code)


def _is_holiday(day: pd.Timestamp, ccaa: str) -> int:
    code = CCAA_HOLIDAY_CODES.get(ccaa)
    if code is None or pd.isna(day):
        return 0
    return int(day.date() in _holiday_calendar(int(day.year), code))


def add_calendar_features(
    df: pd.DataFrame, *, date_col: str = "acq_date"
) -> pd.DataFrame:
    """Add ``month``, ``day_of_year`` and ``es_fin_semana`` from ``date_col``."""
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    out[date_col] = dates
    out["month"] = dates.dt.month
    out["day_of_year"] = dates.dt.dayofyear
    out["es_fin_semana"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
    return out


def add_anthropogenic_features(
    df: pd.DataFrame,
    *,
    date_col: str = "acq_date",
    ccaa_col: str = "ccaa",
    es_fin_semana_col: str = "es_fin_semana",
) -> pd.DataFrame:
    """Add ``es_festivo`` and ``riesgo_antropico_alto`` (weekend OR holiday).

    Requires ``ccaa_col`` and, unless already present, computes the weekend
    flag from ``date_col``.
    """
    out = df.copy()
    if es_fin_semana_col not in out.columns:
        out = add_calendar_features(out, date_col=date_col)
    pairs = out[[date_col, ccaa_col]].drop_duplicates()
    pairs["es_festivo"] = [
        _is_holiday(day, ccaa) for day, ccaa in zip(pairs[date_col], pairs[ccaa_col])
    ]
    out = out.merge(pairs, on=[date_col, ccaa_col], how="left")
    out["es_festivo"] = out["es_festivo"].fillna(0).astype(int)
    out["riesgo_antropico_alto"] = (
        (out[es_fin_semana_col] == 1) | (out["es_festivo"] == 1)
    ).astype(int)
    return out


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos encodings of ``month`` and ``day_of_year`` (period 365.25)."""
    out = df.copy()
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out


def add_budget_features(
    df: pd.DataFrame, *, lookback_years: int | None = None
) -> pd.DataFrame:
    """Add MITECO hectares, prevention/extinction ratio and euros-per-ha.

    Uses the ``prevention_prev{N}`` / ``extinction_prev{N}`` columns already
    joined by :func:`wildfire.data.ccaa.add_ccaa_budget_sums`.
    """
    if lookback_years is None:
        lookback_years = int(load_config()["ccaa"]["budget_lookback_years"])
    prev = f"prevention_prev{lookback_years}"
    ext = f"extinction_prev{lookback_years}"
    missing = [c for c in ("ccaa", prev, ext) if c not in df.columns]
    if missing:
        raise KeyError(f"add_budget_features needs columns: {missing}")

    out = df.copy()
    out["sup_forestal_ha"] = out["ccaa"].map(SUP_FORESTAL_HA)
    out["ratio_prevencion_extincion"] = out[prev] / (out[ext] + 1e-5)
    out["euros_prevencion_por_ha"] = (out[prev] * 1_000_000) / out["sup_forestal_ha"]
    out["euros_extincion_por_ha"] = (out[ext] * 1_000_000) / out["sup_forestal_ha"]
    return out


def add_meteo_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 4 derived fire-weather indices (notebook 01b, cell 2.10)."""
    required = (
        "vapour_pressure_deficit",
        "wind_speed_10m",
        "soil_moisture_0_to_7cm",
        "temperature_2m",
        "relative_humidity_2m",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"add_meteo_indices needs columns: {missing}")

    out = df.copy()
    out["potencial_propagacion"] = (
        out["vapour_pressure_deficit"] * out["wind_speed_10m"]
    )
    out["indice_riesgo_meteo"] = out["potencial_propagacion"] / (
        out["soil_moisture_0_to_7cm"] + 0.01
    )
    out["estres_termico_vegetacion"] = out["temperature_2m"] / (
        out["relative_humidity_2m"] + 1
    )
    out["indice_riesgo_meteo_log"] = np.log1p(out["indice_riesgo_meteo"])
    return out


def add_engineered_features(
    df: pd.DataFrame,
    *,
    lookback_years: int | None = None,
    date_col: str = "acq_date",
) -> pd.DataFrame:
    """Apply every engineering block (calendar → holidays → cyclic → budgets → meteo).

    Assumes weather, CLC and the budget sums are already attached.
    """
    out = add_calendar_features(df, date_col=date_col)
    out = add_anthropogenic_features(out, date_col=date_col)
    out = add_cyclic_features(out)
    out = add_budget_features(out, lookback_years=lookback_years)
    out = add_meteo_indices(out)
    return out


__all__ = [
    "CCAA_HOLIDAY_CODES",
    "SUP_FORESTAL_HA",
    "add_anthropogenic_features",
    "add_budget_features",
    "add_calendar_features",
    "add_cyclic_features",
    "add_engineered_features",
    "add_meteo_indices",
]
