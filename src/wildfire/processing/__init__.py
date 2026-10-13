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
from .occurrence import (
    EVERYDAY_FEATURES,
    SATELLITE_BLOCKLIST,
    SUBDAILY_DROP,
    BlockedSplit,
    add_season,
    assemble_modeling_frame,
    background_skeleton,
    blocked_splits,
    occurrence_features,
    occurrence_settings,
    rank_metrics,
)
from .pca_weather import (
    WeatherPCAResult,
    build_weather_pca_pipeline,
    fit_weather_pca,
    load_target_variance,
    pca_feature_columns,
    pca_input_columns,
    select_n_components,
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
    "EVERYDAY_FEATURES",
    "KEPT",
    "SATELLITE_BLOCKLIST",
    "SUBDAILY_DROP",
    "BlockedSplit",
    "WeatherPCAResult",
    "add_season",
    "add_unified_confidence",
    "apply_modeling_schema",
    "assemble_modeling_frame",
    "background_skeleton",
    "blocked_splits",
    "build_weather_pca_pipeline",
    "categorical_to_numerical",
    "collapse_cross_sensor_dupes",
    "fit_weather_pca",
    "load_target_variance",
    "numerical_to_categorical",
    "occurrence_features",
    "occurrence_settings",
    "pca_feature_columns",
    "pca_input_columns",
    "rank_metrics",
    "select_n_components",
    "validate_enriched",
    "validate_enriched_ccaa",
    "validate_enriched_clc",
    "validate_enriched_openmeteo",
    "validate_firms",
]
