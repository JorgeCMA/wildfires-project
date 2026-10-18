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
    "RISK_BAND_LABELS",
    "MarginCalibration",
    "ablation_verdict",
    "add_risk_bands",
    "apply_isotonic",
    "bootstrap_mean_ci",
    "calibrate_exceedance",
    "calibration_metrics",
    "categorical_columns",
    "categorical_to_dummies",
    "ccaa_day_fire_rates",
    "continuous_columns",
    "cross_fitted_isotonic",
    "dedupe_events",
    "exceedance_base_rate",
    "exceedance_summary",
    "fit_isotonic",
    "fold_scaler_apply",
    "fold_scaler_fit",
    "join_ccaa_fire_rates",
    "margin_positive_labels",
    "model_candidates",
    "prior_correct",
    "quantile_bands",
    "reliability_curve",
]
