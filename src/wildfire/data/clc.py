"""CLCPlus Backbone data loading utilities."""

from __future__ import annotations

from pathlib import Path

from wildfire.config import load_config


def _clc_root(country: str = "Spain") -> Path:
    """Return the root directory for CLCPlus tiles of a country."""
    from wildfire.config import PROJECT_ROOT
    config = load_config()
    return PROJECT_ROOT / config["data"]["raw"] / "clcplus" / country


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
    return sorted(tile_dir.glob("**/*.tif"))



