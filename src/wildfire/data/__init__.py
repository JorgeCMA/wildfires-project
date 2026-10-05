"""Data loading modules."""

from .ccaa import add_ccaa_budget_sums, load_ccaa_budget
from .clc import list_clc_tiles
from .firms import list_available_firms, load_all_firms, load_firms
from .openmeteo import fetch_weather

__all__ = [
    "add_ccaa_budget_sums",
    "fetch_weather",
    "list_available_firms",
    "list_clc_tiles",
    "load_all_firms",
    "load_ccaa_budget",
    "load_firms",
]
