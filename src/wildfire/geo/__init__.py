"""Geospatial helpers (CCAA boundaries)."""

from .regions import REGION_NAME_MAP, assign_ccaa, load_boundaries

__all__ = [
    "REGION_NAME_MAP",
    "assign_ccaa",
    "load_boundaries",
]
