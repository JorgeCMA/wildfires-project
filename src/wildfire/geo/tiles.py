"""Geospatial utilities for programmatic tile selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import Point, box

from wildfire.data.clc import list_clc_tiles


def _tile_bounds(tile_path: Path) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) for a GeoTIFF tile."""
    with rasterio.open(tile_path) as src:
        bounds = src.bounds
    return (bounds.left, bounds.bottom, bounds.right, bounds.top)


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
    point = Point(longitude, latitude)
    tiles = list_clc_tiles(country=country, validity=validity)

    for tile in tiles:
        min_lon, min_lat, max_lon, max_lat = _tile_bounds(tile)
        tile_box = box(min_lon, min_lat, max_lon, max_lat)
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
    with rasterio.open(tile_path) as src:
        transform = src.transform
        row, col = transform.rowcol(longitude, latitude)

    if row < 0 or col < 0:
        raise ValueError(
            f"Coordinates ({latitude}, {longitude}) map to negative pixel indices "
            f"in tile {tile_path.name}"
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
