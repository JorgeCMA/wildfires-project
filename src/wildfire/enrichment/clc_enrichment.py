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

# Neighborhood ring definitions.
# Each ring maps to a list of (row_offset, col_offset, column_suffix) tuples.
# Ring 1: cardinal directions (distance 1)
# Ring 2: diagonals (distance sqrt(2))
# Ring 3: distance-2 cardinal directions
NEIGHBORHOOD_RINGS: dict[int, list[tuple[int, int, str]]] = {
    1: [
        (-1,  0, "N"),
        ( 1,  0, "S"),
        ( 0, -1, "W"),
        ( 0,  1, "E"),
    ],
    2: [
        (-1, -1, "NW"),
        (-1,  1, "NE"),
        ( 1, -1, "SW"),
        ( 1,  1, "SE"),
    ],
    3: [
        (-2,  0, "N2"),
        ( 2,  0, "S2"),
        ( 0, -2, "W2"),
        ( 0,  2, "E2"),
    ],
}


def _resolve_neighbor(
    tile_key: str,
    row: int,
    col: int,
    dr: int,
    dc: int,
    tile_height: int = 10000,
    tile_width: int = 10000,
) -> tuple[str, int, int]:
    """Compute the tile key and local (row, col) for a neighboring pixel.

    Handles cross-tile boundaries by wrapping coordinates and adjusting
    the tile key.  CLCPlus tiles are 10,000 × 10,000 pixels (0-indexed),
    with row 0 at the northern edge and col 0 at the western edge.

    Uses modular arithmetic so the logic is correct for any tile size
    and any ring offset (±1, ±2, etc.).

    Returns
    -------
    (new_tile_key, new_row, new_col)
    """
    raw_row = row + dr
    raw_col = col + dc

    e = int(tile_key[1:tile_key.index("N")])
    n = int(tile_key[tile_key.index("N") + 1:])

    if raw_row < 0:
        n -= 1
    elif raw_row >= tile_height:
        n += 1

    if raw_col < 0:
        e -= 1
    elif raw_col >= tile_width:
        e += 1

    new_row = raw_row % tile_height
    new_col = raw_col % tile_width

    return f"E{e}N{n}", new_row, new_col


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
    """Read a single pixel, returning None for out-of-bounds or nodata."""
    if src is None:
        return None
    if row < 0 or row >= tile_height or col < 0 or col >= tile_width:
        return None
    val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0]
    if np.isnan(val):
        return None
    if src.nodata is not None and val == src.nodata:
        return None
    return int(val)


def enrich_with_clc(
    df: pd.DataFrame,
    country: str = "Spain",
    validity: str = "2023-2025",
) -> pd.DataFrame:
    """Add CLCPlus land cover class and optional neighborhood data.

    For each fire detection the center pixel is read from the corresponding
    CLCPlus tile.  If ``clcplus.neighborhood.enabled`` is ``true`` in the
    project config, neighboring pixels are also read according to the
    configured ``rings`` (see ``NEIGHBORHOOD_RINGS``).

    Added columns
    --------------
    clc_class               Numeric CLCPlus code (center pixel).
    clc_class_{suffix}      Numeric CLCPlus code for each neighbor direction.

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
        DataFrame with center-pixel CLC columns and, if enabled, neighbor
        CLC columns for each configured ring.
    """
    df = df.copy()

    config = load_config()
    clc_config = config.get("clcplus", {})
    neighborhood = clc_config.get("neighborhood", {})
    enabled = neighborhood.get("enabled", False)
    rings = neighborhood.get("rings", []) if enabled else []

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    tile_index = _build_tile_index(country=country, validity=validity)

    xs, ys = transformer.transform(df["longitude"].values, df["latitude"].values)
    tile_keys = [f"E{int(x // 100000)}N{int(y // 100000)}" for x, y in zip(xs, ys)]

    # Collect neighbor offsets for all enabled rings.
    neighbor_offsets: list[tuple[int, int, str]] = []
    for ring in rings:
        if ring in NEIGHBORHOOD_RINGS:
            neighbor_offsets.extend(NEIGHBORHOOD_RINGS[ring])

    # Prepare storage for center pixel.
    clc_classes: list[int | None] = []

    # Prepare storage for neighbor columns.
    neigh_classes: dict[str, list[int | None]] = {
        f"clc_class_{s}": [] for _, _, s in neighbor_offsets
    }

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
            for key in neigh_classes:
                neigh_classes[key].append(None)
            continue

        row, col = reader.index(x, y)
        height, width = _get_dims(tk)
        val = _pixel_value(reader, row, col, width, height)

        clc_classes.append(val)

        # Read neighbor pixels.
        for dr, dc, suffix in neighbor_offsets:
            ntk, nr, nc = _resolve_neighbor(
                tk, row, col, dr, dc,
                tile_height=height, tile_width=width,
            )
            n_reader = _get_reader(ntk)
            n_height, n_width = _get_dims(ntk)
            n_val = _pixel_value(n_reader, nr, nc, n_width, n_height)
            neigh_classes[f"clc_class_{suffix}"].append(n_val)

    for reader in open_readers.values():
        if reader is not None:
            reader.close()

    df["clc_class"] = clc_classes

    for key in neigh_classes:
        df[key] = neigh_classes[key]

    # Boolean: true if all 4 cardinal neighbors match the center pixel.
    # Requires center pixel to be non-null (avoids None == None → True).
    cardinal_suffixes = [s for _, _, s in NEIGHBORHOOD_RINGS.get(1, [])]
    cardinal_cols = [f"clc_class_{s}" for s in cardinal_suffixes]
    if cardinal_cols and all(c in df.columns for c in cardinal_cols):
        center_valid = df["clc_class"].notna()
        all_match = center_valid.copy()
        for col_name in cardinal_cols:
            all_match &= df[col_name].notna() & (df["clc_class"] == df[col_name])
        df["clc_uniform_surroundings"] = all_match
    else:
        df["clc_uniform_surroundings"] = False

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
