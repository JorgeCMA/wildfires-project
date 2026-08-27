"""CLCPlus Backbone data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from wildfire.config import load_config


def _clc_root(country: str = "Spain") -> Path:
    """Return the root directory for CLCPlus tiles of a country."""
    config = load_config()
    return Path(config["data"]["raw"]) / "clcplus" / country


def list_clc_tiles(country: str = "Spain", validity: str = "2023-2025") -> list[Path]:
    """List all GeoTIFF tiles for a given country and validity period.

    Parameters
    ----------
    country:
        Country folder name.
    validity:
        Validity period folder name (e.g. ``"2023-2025"``).

    Returns
    -------
    list[Path]
        Sorted list of ``.tif`` file paths.
    """
    tile_dir = _clc_root(country) / validity
    if not tile_dir.exists():
        return []
    return sorted(tile_dir.glob("*.tif"))


def load_clc_tile(
    tile_path: Path | str,
    window: Window | None = None,
) -> tuple[np.ndarray, dict]:
    """Load a single CLCPlus GeoTIFF tile.

    Parameters
    ----------
    tile_path:
        Path to the ``.tif`` file.
    window:
        Optional rasterio Window to read only a portion of the raster.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple of (data array, metadata dict with keys ``transform``,
        ``crs``, ``bounds``, ``shape``).
    """
    tile_path = Path(tile_path)
    if not tile_path.exists():
        raise FileNotFoundError(f"CLCPlus tile not found: {tile_path}")

    with rasterio.open(tile_path) as src:
        if window:
            data = src.read(1, window=window)
        else:
            data = src.read(1)

        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
            "shape": data.shape,
            "nodata": src.nodata,
            "resolutions": src.res,
        }

    return data, meta


def read_pixel_value(
    tile_path: Path | str,
    row: int,
    col: int,
) -> int | float | None:
    """Read a single pixel value from a CLCPlus tile.

    Parameters
    ----------
    tile_path:
        Path to the ``.tif`` file.
    row:
        Row index in the raster grid.
    col:
        Column index in the raster grid.

    Returns
    -------
    int | float | None
        The pixel value, or ``None`` if it equals the nodata value.
    """
    tile_path = Path(tile_path)
    with rasterio.open(tile_path) as src:
        data = src.read(1, window=Window(col, row, 1, 1))
        value = data[0, 0]
        if src.nodata is not None and value == src.nodata:
            return None
        return int(value) if value == int(value) else float(value)
