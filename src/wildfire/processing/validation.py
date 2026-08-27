"""Data quality validation for FIRMS and enriched datasets."""

from __future__ import annotations

import pandas as pd


def validate_firms(df: pd.DataFrame) -> list[str]:
    """Run basic quality checks on a FIRMS DataFrame.

    Returns a list of warning messages. An empty list means all checks passed.

    Parameters
    ----------
    df:
        FIRMS DataFrame (merged or raw).

    Returns
    -------
    list[str]
        Validation warnings.
    """
    warnings: list[str] = []

    required = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    if "latitude" in df.columns:
        lat = df["latitude"]
        out_of_range = (lat < -90) | (lat > 90)
        if out_of_range.any():
            n = int(out_of_range.sum())
            warnings.append(f"{n} rows have latitude outside [-90, 90]")

    if "longitude" in df.columns:
        lon = df["longitude"]
        out_of_range = (lon < -180) | (lon > 180)
        if out_of_range.any():
            n = int(out_of_range.sum())
            warnings.append(f"{n} rows have longitude outside [-180, 180]")

    if "frp" in df.columns:
        neg_frp = (df["frp"] < 0).sum()
        if neg_frp > 0:
            warnings.append(f"{neg_frp} rows have negative FRP values")

    if "acq_date" in df.columns:
        try:
            pd.to_datetime(df["acq_date"])
        except Exception:
            warnings.append("acq_date column could not be parsed as datetime")

    if df.empty:
        warnings.append("DataFrame is empty")

    return warnings


def validate_enriched(df: pd.DataFrame) -> list[str]:
    """Run quality checks on an enriched FIRMS DataFrame.

    Extends ``validate_firms`` with checks for enrichment columns.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.

    Returns
    -------
    list[str]
        Validation warnings.
    """
    warnings = validate_firms(df)

    clc_cols = ["clc_class"]
    for col in clc_cols:
        if col in df.columns and df[col].isna().any():
            n = int(df[col].isna().sum())
            warnings.append(f"{n} rows have missing {col} values (CLC enrichment gap)")

    weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
    for col in weather_cols:
        if col in df.columns and df[col].isna().any():
            n = int(df[col].isna().sum())
            warnings.append(f"{n} rows have missing {col} values (weather enrichment gap)")

    return warnings
