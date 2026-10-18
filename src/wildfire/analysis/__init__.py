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
    "CANDIDATE_CATEGORICAL",
    "CANDIDATE_CONTINUOUS",
    "CATEGORICAL_FEATURES",
    "CONTINUOUS_FEATURES",
    "DEFAULT_CATEGORICAL",
    "DEFAULT_MARGIN",
    "DEFAULT_N_BOOT",
    "ablation_verdict",
    "bootstrap_mean_ci",
    "categorical_columns",
    "categorical_to_dummies",
    "continuous_columns",
    "fold_scaler_apply",
    "fold_scaler_fit",
    "model_candidates",
]
