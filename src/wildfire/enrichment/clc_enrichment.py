"""Enrich FIRMS data with CLCPlus Backbone land cover classification."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from wildfire.config import load_config
from wildfire.data.clc import list_clc_tiles


def load_clc_classes(
    country: str = "Spain",
    validity: str = "2023-2025",
) -> pd.DataFrame:
    """Read CLCPlus class definitions from tile XML sidecar files.

    Each ``.tif`` has a ``.tif.aux.xml`` with a ``GDALRasterAttributeTable``
    listing value, count, Class_name, Area_km2, and Area_perc.  This
    function reads the first available XML and returns a DataFrame with
    one row per class.

    Parameters
    ----------
    country:
        Country folder name.
    validity:
        Validity period folder name.

    Returns
    -------
    pd.DataFrame
        Columns: ``clc_code``, ``clc_name``, ``pixel_count``, ``area_km2``,
        ``area_perc``.
    """
    tiles = list_clc_tiles(country=country, validity=validity)
    for tile in tiles:
        xml_path = tile.with_suffix(".tif.aux.xml")
        if not xml_path.exists():
            continue

        tree = ElementTree.parse(xml_path)
        rows = tree.findall(".//Row")
        records = []
        for row in rows:
            fields = [f.text for f in row.findall("F")]
            records.append({
                "clc_code": int(fields[0]),
                "clc_name": fields[2],
                "pixel_count": int(fields[1]),
                "area_km2": float(fields[3]),
                "area_perc": float(fields[4]),
            })
        df = pd.DataFrame(records).sort_values("clc_code").reset_index(drop=True)
        return df

    return pd.DataFrame(columns=["clc_code", "clc_name", "pixel_count", "area_km2", "area_perc"])


# CLCPlus Backbone class labels.
# Source: .tif.aux.xml GDALRasterAttributeTable from CLCPlus tiles.
CLC_LABELS: dict[int, str] = {
    1: "Sealed",
    2: "Woody needle leaved trees",
    3: "Woody broadleaved deciduous trees",
    4: "Woody broadleaved evergreen trees",
    5: "Low-growing woody plants",
    6: "Permanent herbaceous",
    7: "Periodically herbaceous",
    8: "Lichens and mosses",
    9: "Non and sparsely vegetated",
    10: "Water",
    11: "Snow and ice",
    253: "Coastal seawater buffer",
    254: "Outside area",
    255: "No data",
}

# Neighbor offsets: (row_offset, col_offset) for T, TR, R, BR, B, BL, L, TL
_NEIGHBOR_OFFSETS = {
    "clc_T":  (-1,  0),
    "clc_TR": (-1,  1),
    "clc_R":  ( 0,  1),
    "clc_BR": ( 1,  1),
    "clc_B":  ( 1,  0),
    "clc_BL": ( 1, -1),
    "clc_L":  ( 0, -1),
    "clc_TL": (-1, -1),
}

CLC_NEIGHBOR_COLS = ["clc_C"] + list(_NEIGHBOR_OFFSETS.keys())


def _build_tile_index(
    country: str = "Spain",
    validity: str = "2023-2025",
) -> dict[str, Path]:
    """Build a dict mapping tile keys (e.g. 'E31N23') to .tif paths."""
    tiles = list_clc_tiles(country=country, validity=validity)
    index: dict[str, Path] = {}
    for tile in tiles:
        # Extract E{XX}N{YY} from filename like
        # CLMS_CLCPLUS_RAS_S2023_R10m_E31N23_03035_V01_R00
        parts = tile.stem.split("_")
        for part in parts:
            if part.startswith("E") and "N" in part:
                index[part] = tile
                break
    return index


def _pixel_value(
    src: rasterio.DatasetReader | None,
    row: int,
    col: int,
    tile_width: int,
    tile_height: int,
) -> int | None:
    """Read a single pixel, returning None for out-of-bounds."""
    if src is None:
        return None
    if row < 0 or row >= tile_height or col < 0 or col >= tile_width:
        return None
    val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
    if src.nodata is not None and val == src.nodata:
        return None
    return int(val) if val == int(val) else float(val)


def _extract_tile_key(stem: str) -> str | None:
    """Extract the tile key (e.g. 'E31N23') from a GeoTIFF filename stem."""
    for part in stem.split("_"):
        if part.startswith("E") and "N" in part:
            return part
    return None


def enrich_with_clc(
    df: pd.DataFrame,
    country: str = "Spain",
    validity: str = "2023-2025",
) -> pd.DataFrame:
    """Add CLCPlus land cover class to each FIRMS row.

    For each fire detection the center pixel is read from the corresponding
    CLCPlus tile.

    Added columns
    --------------
    clc_class    Numeric CLCPlus code.
    clc_name     Human-readable label.

    Parameters
    ----------
    df:
        FIRMS DataFrame with ``latitude`` and ``longitude`` columns.
    country:
        Country folder name for CLCPlus tiles.
    validity:
        Validity period folder name.

    Returns
    -------
    pd.DataFrame
        DataFrame with the ``clc_class`` and ``clc_name`` columns.
    """
    df = df.copy()

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    tile_index = _build_tile_index(country=country, validity=validity)

    xs, ys = transformer.transform(df["longitude"].values, df["latitude"].values)
    tile_keys = [f"E{int(x // 100000)}N{int(y // 100000)}" for x, y in zip(xs, ys)]

    clc_classes: list[int | None] = []
    clc_names: list[str | None] = []

    open_readers: dict[str, rasterio.DatasetReader | None] = {}
    tile_dims: dict[str, tuple[int, int]] = {}

    def _get_reader(tk: str) -> rasterio.DatasetReader | None:
        if tk not in open_readers:
            path = tile_index.get(tk)
            open_readers[tk] = rasterio.open(path) if path else None
        return open_readers[tk]

    def _get_dims(tk: str) -> tuple[int, int]:
        if tk not in tile_dims:
            reader = _get_reader(tk)
            tile_dims[tk] = (reader.height, reader.width) if reader else (0, 0)
        return tile_dims[tk]

    for idx, (x, y, tk) in enumerate(zip(xs, ys, tile_keys)):
        reader = _get_reader(tk)
        if reader is None:
            clc_classes.append(None)
            clc_names.append(None)
            continue

        row, col = reader.index(x, y)
        height, width = _get_dims(tk)
        val = _pixel_value(reader, row, col, width, height)

        clc_classes.append(val)
        clc_names.append(CLC_LABELS.get(val, f"Unknown ({val})") if val is not None else None)

    for reader in open_readers.values():
        if reader is not None:
            reader.close()

    df["clc_class"] = clc_classes
    df["clc_name"] = clc_names
    return df


def save_enriched(df: pd.DataFrame, country: str = "Spain", year: int | None = None) -> Path:
    """Save enriched FIRMS data to ``data/processed/enriched/``.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.
    country:
        Country name.
    year:
        If provided, save as ``firms_{country}_{year}_enriched.csv``.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    config = load_config()
    out_dir = Path(config["output"]["enriched"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_enriched.csv"
    else:
        filename = f"firms_{country.lower()}_enriched.csv"

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def load_enriched(country: str = "Spain", year: int | None = None) -> pd.DataFrame:
    """Load previously enriched FIRMS data.

    Parameters
    ----------
    country:
        Country name.
    year:
        If provided, loads ``firms_{country}_{year}_enriched.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the enriched CSV file does not exist.
    """
    config = load_config()
    out_dir = Path(config["output"]["enriched"])

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_enriched.csv"
    else:
        filename = f"firms_{country.lower()}_enriched.csv"

    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Enriched file not found: {path}")

    return pd.read_csv(path)
