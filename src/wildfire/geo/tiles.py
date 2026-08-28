"""Geospatial utilities for programmatic tile selection."""

from __future__ import annotations

from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol as rio_rowcol
from shapely.geometry import Point, box

from wildfire.data.clc import list_clc_tiles

# Shared transformer: WGS84 (EPSG:4326) → ETRS89-LAEA (EPSG:3035)
_TRANSFORMER_TO_3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)


def find_tile_for_point(
    latitude: float,
    longitude: float,
    country: str = "Spain",
    validity: str = "2023-2025",
) -> Path | None:
    """Find the CLCPlus tile that contains a given geographic point.

    Parameters
    ----------
    latitude:
        Latitude in decimal degrees.
    longitude:
        Longitude in decimal degrees.
    country:
        Country folder name.
    validity:
        Validity period folder name.

    Returns
    -------
    Path | None
        Path to the matching tile, or ``None`` if no tile contains the point.
    """
    x, y = _TRANSFORMER_TO_3035.transform(longitude, latitude)
    point = Point(x, y)
    tiles = list_clc_tiles(country=country, validity=validity)

    for tile in tiles:
        with rasterio.open(tile) as src:
            tile_box = box(*src.bounds)
            if tile_box.contains(point):
                return tile

    return None


def latlon_to_pixel(
    tile_path: Path,
    latitude: float,
    longitude: float,
) -> tuple[int, int]:
    """Convert latitude/longitude to row/col pixel indices in a GeoTIFF tile.

    Parameters
    ----------
    tile_path:
        Path to the GeoTIFF tile.
    latitude:
        Latitude in decimal degrees.
    longitude:
        Longitude in decimal degrees.

    Returns
    -------
    tuple[int, int]
        (row, col) pixel indices.

    Raises
    ------
    ValueError
        If the coordinates fall outside the tile bounds.
    """
    x, y = _TRANSFORMER_TO_3035.transform(longitude, latitude)

    with rasterio.open(tile_path) as src:
        row, col = rio_rowcol(src.transform, x, y)

    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise ValueError(
            f"Coordinates ({latitude}, {longitude}) map to pixel "
            f"({row}, {col}) outside tile {tile_path.name} "
            f"(size {src.height}x{src.width})"
        )

    return int(row), int(col)


def find_tile_and_pixel(
    latitude: float,
    longitude: float,
    country: str = "Spain",
    validity: str = "2023-2025",
) -> tuple[Path, int, int] | None:
    """Find the tile and pixel coordinates for a geographic point.

    Parameters
    ----------
    latitude:
        Latitude in decimal degrees.
    longitude:
        Longitude in decimal degrees.
    country:
        Country folder name.
    validity:
        Validity period folder name.

    Returns
    -------
    tuple[Path, int, int] | None
        (tile_path, row, col) or ``None`` if no tile contains the point.
    """
    tile = find_tile_for_point(latitude, longitude, country, validity)
    if tile is None:
        return None
    row, col = latlon_to_pixel(tile, latitude, longitude)
    return tile, row, col
