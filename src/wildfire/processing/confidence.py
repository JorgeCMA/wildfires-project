"""Confidence mapping between MODIS numerical and VIIRS categorical values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from wildfire.config import load_config
from wildfire.data.firms import load_all_firms


def _load_thresholds() -> dict:
    """Load the confidence thresholds YAML file."""
    config = load_config()
    path = Path(config["paths"]["confidence_thresholds"])
    with open(path) as f:
        return yaml.safe_load(f)


def numerical_to_categorical(value: float, thresholds: dict | None = None) -> str:
    """Convert a MODIS numerical confidence (0-100) to a categorical label.

    Parameters
    ----------
    value:
        Numerical confidence value.
    thresholds:
        Optional pre-loaded thresholds dict. Loaded from YAML if ``None``.

    Returns
    -------
    str
        One of ``"low"``, ``"nominal"``, or ``"high"``.
    """
    if thresholds is None:
        thresholds = _load_thresholds()
    ranges = thresholds["numerical_to_categorical"]

    if pd.isna(value):
        return "nominal"

    for label in ("low", "nominal", "high"):
        if ranges[label]["min"] <= value < ranges[label]["max"]:
            return label[0]  # l, n, or h
    return "h"


def categorical_to_numerical(value: str, thresholds: dict | None = None) -> float:
    """Convert a VIIRS categorical confidence (l/n/h) to a numerical value.

    Parameters
    ----------
    value:
        One of ``"l"``, ``"n"``, ``"h"`` (case-insensitive).
    thresholds:
        Optional pre-loaded thresholds dict.

    Returns
    -------
    float
        The representative numerical value.
    """
    if thresholds is None:
        thresholds = _load_thresholds()
    mapping = thresholds["categorical_to_numerical"]

    if pd.isna(value):
        return mapping["n"]

    value_lower = str(value).strip().lower()
    return mapping.get(value_lower, mapping["n"])


def add_unified_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """Add unified ``confidence_cat`` and ``confidence_num`` columns.

    This function expects a merged FIRMS DataFrame with columns:
    - ``sensor``: ``"modis"``, ``"viirs_snpp"``, or ``"viirs_noaa20"``
    - ``confidence``: original confidence value (numerical for MODIS, categorical for VIIRS)

    It produces:
    - ``confidence_og_num``: original numerical value (from MODIS, ``NaN`` for VIIRS)
    - ``confidence_og_cat``: original categorical value (from VIIRS, ``NaN`` for MODIS)
    - ``confidence_cat``: unified categorical label
    - ``confidence_num``: unified numerical value

    Parameters
    ----------
    df:
        Merged FIRMS DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with the additional confidence columns.
    """
    df = df.copy()
    thresholds = _load_thresholds()

    df["confidence_og_num"] = pd.NA
    df["confidence_og_cat"] = pd.NA

    modis_mask = df["sensor"] == "modis"
    viirs_mask = df["sensor"].str.startswith("viirs")

    df.loc[modis_mask, "confidence_og_num"] = df.loc[modis_mask, "confidence"]
    df.loc[viirs_mask, "confidence_og_cat"] = df.loc[viirs_mask, "confidence"]

    df["confidence_num"] = pd.NA
    df["confidence_cat"] = pd.NA

    # MODIS: numerical → categorical
    df.loc[modis_mask, "confidence_num"] = pd.to_numeric(
        df.loc[modis_mask, "confidence"], errors="coerce"
    )
    df.loc[modis_mask, "confidence_cat"] = df.loc[modis_mask, "confidence_num"].apply(
        lambda v: numerical_to_categorical(v, thresholds)
    )

    # VIIRS: categorical → numerical
    df.loc[viirs_mask, "confidence_cat"] = df.loc[viirs_mask, "confidence"].str.lower()
    df.loc[viirs_mask, "confidence_num"] = df.loc[viirs_mask, "confidence_cat"].apply(
        lambda v: categorical_to_numerical(v, thresholds)
    )

    df["confidence_cat"] = df["confidence_cat"].astype("category")
    df["confidence_num"] = pd.to_numeric(df["confidence_num"], errors="coerce")

    return df
