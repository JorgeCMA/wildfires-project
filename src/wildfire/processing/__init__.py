"""Processing and validation modules."""

from .confidence import (
    add_unified_confidence,
    categorical_to_numerical,
    numerical_to_categorical,
)
from .dedup_cross_sensor import (
    DROPPED_MODIS_OVERLAP,
    KEPT,
    apply_modeling_schema,
    collapse_cross_sensor_dupes,
)
from .validation import (
    validate_enriched,
    validate_enriched_ccaa,
    validate_enriched_clc,
    validate_enriched_openmeteo,
    validate_firms,
)

__all__ = [
    "DROPPED_MODIS_OVERLAP",
    "KEPT",
    "add_unified_confidence",
    "apply_modeling_schema",
    "categorical_to_numerical",
    "collapse_cross_sensor_dupes",
    "numerical_to_categorical",
    "validate_enriched",
    "validate_enriched_ccaa",
    "validate_enriched_clc",
    "validate_enriched_openmeteo",
    "validate_firms",
]
