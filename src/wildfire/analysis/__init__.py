"""Modeling units for the supervised occurrence model (block 04).

Small, tested helpers the notebook cells call (same architecture rule as the
rest of the project): model candidates, per-fold scaling, fold-fit one-hot
encoding and the ablation verdict. The experiment loop itself lives in
``notebooks/04_aprendizaje_supervisado.ipynb``.
"""

from .ablation import (
    ABLATION_SEED,
    DEFAULT_MARGIN,
    DEFAULT_N_BOOT,
    ablation_verdict,
    bootstrap_mean_ci,
)
from .calibration import (
    EXCEEDANCE_MARGINS_KM,
    MarginCalibration,
    apply_isotonic,
    calibrate_exceedance,
    calibration_metrics,
    cross_fitted_isotonic,
    dedupe_events,
    exceedance_base_rate,
    exceedance_summary,
    fit_isotonic,
    margin_positive_labels,
    prior_correct,
    reliability_curve,
)
from .demo_data import (
    RISK_BAND_COLORS,
    band_counts,
    ccaa_base_rate_geojson,
    load_ccaa_rates,
    load_detections,
    load_fold_metrics,
    load_reliability,
    load_scores,
)
from .deployment import (
    ALL_LABEL,
    RISK_BAND_LABELS,
    add_risk_bands,
    ccaa_day_fire_rates,
    join_ccaa_fire_rates,
    quantile_bands,
)
from .occurrence_model import (
    CANDIDATE_CATEGORICAL,
    CANDIDATE_CONTINUOUS,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    DEFAULT_CATEGORICAL,
    categorical_columns,
    categorical_to_dummies,
    continuous_columns,
    fold_scaler_apply,
    fold_scaler_fit,
    model_candidates,
)
from .status import format_pool_status, pool_status

__all__ = [
    "ABLATION_SEED",
    "ALL_LABEL",
    "CANDIDATE_CATEGORICAL",
    "CANDIDATE_CONTINUOUS",
    "CATEGORICAL_FEATURES",
    "CONTINUOUS_FEATURES",
    "DEFAULT_CATEGORICAL",
    "DEFAULT_MARGIN",
    "DEFAULT_N_BOOT",
    "EXCEEDANCE_MARGINS_KM",
    "RISK_BAND_COLORS",
    "RISK_BAND_LABELS",
    "MarginCalibration",
    "ablation_verdict",
    "add_risk_bands",
    "apply_isotonic",
    "band_counts",
    "bootstrap_mean_ci",
    "calibrate_exceedance",
    "calibration_metrics",
    "categorical_columns",
    "categorical_to_dummies",
    "ccaa_base_rate_geojson",
    "ccaa_day_fire_rates",
    "continuous_columns",
    "cross_fitted_isotonic",
    "dedupe_events",
    "exceedance_base_rate",
    "exceedance_summary",
    "fit_isotonic",
    "fold_scaler_apply",
    "fold_scaler_fit",
    "format_pool_status",
    "join_ccaa_fire_rates",
    "load_ccaa_rates",
    "load_detections",
    "load_fold_metrics",
    "load_reliability",
    "load_scores",
    "margin_positive_labels",
    "model_candidates",
    "pool_status",
    "prior_correct",
    "quantile_bands",
    "reliability_curve",
]
