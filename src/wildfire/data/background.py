"""Synthetic background sample for the future occurrence model.

Generates random (date, lat/lon) rows inside Spanish land territory with the
same 22 columns as ``data/processed/merged/firms_spain_merged.csv``.
Satellital observables are NaN (background points are not detections);
weather, CLC and remaining enrichment are resolved later by the pipeline,
which only needs ``latitude``/``longitude``/``acq_date``/``acq_time``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import contains_xy

from wildfire.data.io import load_csv, save_csv
from wildfire.geo.regions import load_boundaries

MERGED_RELATIVE = "data/processed/merged/firms_spain_merged.csv"
DEFAULT_OUTPUT = "data/processed/synthetic/firms_spain_background_100k.csv"

# Columnas satelitales/derivadas: NaN en el fondo (no son detecciones).
SATELLITAL_NAN_COLS = (
    "brightness",
    "scan",
    "track",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "brightness_ir",
    "frp",
    "sensor",
    "confidence_og_num",
    "confidence_og_cat",
    "confidence_num",
    "confidence_cat",
)

# Columnas estructurales con valor fijo (no satelitales).
COUNTRY = "Spain"
BACKGROUND_TYPE = 0  # como el filtro decidido `type == 0` del pipeline


def _land_geometry() -> gpd.GeoSeries:
    """Dissolved Spanish land geometry (EPSG:4326) from the team GeoJSON."""
    boundaries = load_boundaries()
    return gpd.GeoSeries([boundaries.geometry.union_all()], crs=boundaries.crs)


def _sample_land_points(
    n: int, rng: np.random.Generator, chunk: int = 200_000
) -> tuple[np.ndarray, np.ndarray, float]:
    """Uniform bbox draws rejected against the land geometry.

    Points are rounded to 4 decimals (FIRMS precision) and re-verified after
    rounding, since coastal points can flip with rounding. Uses vectorized
    ``shapely.contains_xy`` (no GeoSeries alignment involved).

    Returns ``(lat, lon, acceptance_rate)``.
    """
    poly = _land_geometry().iloc[0]
    xmin, ymin, xmax, ymax = poly.bounds
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    drawn = 0
    kept = 0
    while kept < n:
        xs = rng.uniform(xmin, xmax, chunk)
        ys = rng.uniform(ymin, ymax, chunk)
        drawn += chunk
        mask = contains_xy(poly, xs, ys)
        rx = np.round(xs[mask], 4)
        ry = np.round(ys[mask], 4)
        ok = contains_xy(poly, rx, ry)
        take = min(int(ok.sum()), n - kept)
        lats.append(ry[ok][:take])
        lons.append(rx[ok][:take])
        kept += take
    lat = np.concatenate(lats)[:n]
    lon = np.concatenate(lons)[:n]
    return lat, lon, kept / drawn


def _sample_overpass_times(
    n: int, rng: np.random.Generator, merged_path: str | Path = MERGED_RELATIVE
) -> tuple[np.ndarray, np.ndarray]:
    """Jointly resample (daynight, acq_time) pairs from the real merged CSV.

    Preserves the observed overpass structure (night mode ~1-3h, day ~12-14h)
    and guarantees HHMM-valid times for the weather batch validator.
    """
    merged = load_csv(merged_path, usecols=["daynight", "acq_time"])
    pairs = merged[["daynight", "acq_time"]].dropna().to_numpy()
    if len(pairs) == 0:
        raise ValueError(f"No (daynight, acq_time) pairs in {merged_path}")
    pick = pairs[rng.integers(0, len(pairs), n)]
    return pick[:, 0], pick[:, 1].astype(int)


def generate_background(
    n: int = 100_000,
    seed: int = 42,
    start: str = "2023-01-01",
    end: str = "2024-12-31",
    output: str | Path = DEFAULT_OUTPUT,
    merged_path: str | Path = MERGED_RELATIVE,
) -> pd.DataFrame:
    """Generate the synthetic background sample and save it to CSV.

    Parameters
    ----------
    n:
        Number of rows.
    seed:
        RNG seed (reproducibility).
    start, end:
        Uniform date window (ISO ``YYYY-MM-DD``, inclusive).
    output:
        Project-root-relative output path.
    merged_path:
        Real merged CSV: fixes the column order and the overpass pool.

    Returns
    -------
    pd.DataFrame
        The background frame (same columns/order as the merged CSV).
    """
    rng = np.random.default_rng(seed)
    reference_cols = list(load_csv(merged_path, nrows=0).columns)

    start_ord = date.fromisoformat(start).toordinal()
    end_ord = date.fromisoformat(end).toordinal()
    days = rng.integers(start_ord, end_ord + 1, n)
    dates = pd.to_datetime([date.fromordinal(int(d)) for d in days])

    lat, lon, acceptance = _sample_land_points(n, rng)
    daynight, acq_time = _sample_overpass_times(n, rng, merged_path)

    df = pd.DataFrame(
        {
            "latitude": np.round(lat, 4),
            "longitude": np.round(lon, 4),
            "acq_date": dates.strftime("%Y-%m-%d"),
            "acq_time": acq_time,
            "daynight": daynight,
            "country": COUNTRY,
            "year": dates.year.to_numpy(),
            "type": np.full(n, BACKGROUND_TYPE, dtype=np.int64),
            **{col: np.nan for col in SATELLITAL_NAN_COLS},
        }
    )[reference_cols]
    _validate_background(df, reference_cols, start, end)

    path = save_csv(df, output)
    print(f"Saved {len(df):,} background rows to {path}")
    print(f"Land acceptance rate: {acceptance:.3f}")
    return df


def _validate_background(
    df: pd.DataFrame, reference_cols: list[str], start: str, end: str
) -> None:
    """Fail-closed contract checks against the merged schema."""
    if list(df.columns) != reference_cols:
        raise ValueError("Background columns differ from the merged CSV")
    for col in ("latitude", "longitude", "acq_date", "acq_time", "country", "year", "type"):
        if df[col].isna().any():
            raise ValueError(f"Structural column {col!r} has NaN")
    if not df["acq_date"].between(start, end).all():
        raise ValueError("acq_date outside the requested window")
    if (df["year"].astype(str) != df["acq_date"].str[:4]).any():
        raise ValueError("year/acq_date mismatch")
    hhmm = df["acq_time"].astype(int)
    if ((hhmm // 100 > 23) | (hhmm % 100 > 59)).any():
        raise ValueError("acq_time is not valid HHMM")
    land = _land_geometry()
    if not bool(
        contains_xy(
            land.iloc[0],
            df["longitude"].to_numpy(),
            df["latitude"].to_numpy(),
        ).all()
    ):
        raise ValueError("Some background points fall outside Spanish land")


if __name__ == "__main__":  # pragma: no cover
    generate_background()


__all__ = [
    "BACKGROUND_TYPE",
    "COUNTRY",
    "DEFAULT_OUTPUT",
    "MERGED_RELATIVE",
    "SATELLITAL_NAN_COLS",
    "generate_background",
]
