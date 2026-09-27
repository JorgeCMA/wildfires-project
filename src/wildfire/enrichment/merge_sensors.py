"""Merge VIIRS and MODIS FIRMS data into a unified dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.config import load_config
from wildfire.data.firms import load_all_firms
from wildfire.processing.confidence import add_unified_confidence
from wildfire.processing.validation import validate_firms


def merge_viirs_modis(
    country: str = "Spain",
    years: list[int] | None = None,
) -> pd.DataFrame:
    """Load VIIRS and MODIS data, merge, and add unified confidence columns.

    Parameters
    ----------
    country:
        Country name.
    years:
        List of years to load. Defaults to configured years.

    Returns
    -------
    pd.DataFrame
        Merged FIRMS data with unified confidence columns.
    """
    df = load_all_firms(country=country, years=years)
    if df.empty:
        return df

    df = add_unified_confidence(df)
    return df


def save_merged(df: pd.DataFrame, country: str = "Spain", year: int | None = None) -> Path:
    """Save merged FIRMS data to ``data/processed/merged/``.

    Parameters
    ----------
    df:
        Merged FIRMS DataFrame.
    country:
        Country name.
    year:
        If provided, save as ``firms_{country}_{year}_merged.csv``.
        Otherwise save as ``firms_{country}_merged.csv``.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    config = load_config()
    out_dir = Path(config["output"]["merged"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_merged.csv"
    else:
        filename = f"firms_{country.lower()}_merged.csv"

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def load_merged(country: str = "Spain", year: int | None = None) -> pd.DataFrame:
    """Load previously merged FIRMS data from ``data/processed/merged/``.

    Parameters
    ----------
    country:
        Country name.
    year:
        If provided, loads ``firms_{country}_{year}_merged.csv``.

    Returns
    -------
    pd.DataFrame
        The merged FIRMS data.

    Raises
    ------
    FileNotFoundError
        If the merged CSV file does not exist.
    """
    config = load_config()
    out_dir = Path(config["output"]["merged"])

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_merged.csv"
    else:
        filename = f"firms_{country.lower()}_merged.csv"

    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Merged file not found: {path}")

    return pd.read_csv(path)


def load_latest_merged_split(
    country: str = "Spain",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recently modified merged CSV and split it in two parts.

    Parameters
    ----------
    country:
        Country name used to filter candidate files.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        The first 3 rows of the latest merged CSV and all remaining rows.

    Raises
    ------
    FileNotFoundError
        If no merged CSV exists for the country in ``data/processed/merged/``.
    """
    config = load_config()
    out_dir = Path(config["output"]["merged"])

    candidates = list(out_dir.glob(f"firms_{country.lower()}*_merged.csv"))
    if not candidates:
        raise FileNotFoundError(f"No merged files found in: {out_dir}")

    path = max(candidates, key=lambda p: p.stat().st_mtime)
    df = pd.read_csv(path)
    return df.head(3), df.iloc[3:]
