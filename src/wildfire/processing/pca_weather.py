"""PCA over raw weather variables (notebook 02, exploratory + reusable).

The PCA input is 14 continuous columns: 12 raw weather variables plus the
sine/cosine expansion of the circular ``wind_direction_10m``. Excluded on
purpose (see notebook discussion):

- ``rain``: identical to ``precipitation`` (corr = 1.0) — keep one.
- ``is_day``: binary recode of ``daynight`` — not a continuous PCA input.
- ``wind_direction_10m`` raw: circular (359 ~ 1); enters only as
  ``wd_sin``/``wd_cos`` (unweighted; wind speed already enters on its own).
- Derived indices (``potencial_propagacion``, ``indice_riesgo_meteo[_log]``,
  ``estres_termico_vegetacion``, ``indice_meteorologico``): built from the
  raw variables — correlate them with the PCs afterwards instead.

Preprocessing (all inside the pipeline, so a later train/test split cannot
leak): median imputation, ``log1p`` on ``precipitation`` only (skew ~25,
~95% zeros), z-score standardization, full PCA. The number of retained
components ``k`` comes from ``pca.target_variance`` in
``configs/project.yaml`` (fraction of cumulative variance, 80-90% rule +
scree inspection in the notebook).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from wildfire.config import load_config

WIND_DIRECTION_COL = "wind_direction_10m"
PRECIPITATION_COL = "precipitation"
WIND_SIN_COL = "wd_sin"
WIND_COS_COL = "wd_cos"

# 12 continuous raw weather variables (no rain / is_day / raw direction).
WEATHER_PCA_BASE: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "shortwave_radiation",
    "boundary_layer_height",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
]


def pca_input_columns() -> list[str]:
    """Raw columns the pipeline expects (12 base + circular direction)."""
    return [*WEATHER_PCA_BASE, WIND_DIRECTION_COL]


def pca_feature_columns() -> list[str]:
    """Model features in fixed order (12 base + wd_sin/wd_cos = 14)."""
    return [*WEATHER_PCA_BASE, WIND_SIN_COL, WIND_COS_COL]


def load_target_variance() -> float:
    """Target cumulative-variance fraction from ``pca.target_variance``."""
    try:
        target = float(load_config()["pca"]["target_variance"])
    except KeyError as exc:
        raise KeyError("Missing 'pca.target_variance' in configs/project.yaml") from exc
    _validate_target_variance(target)
    return target


def _validate_target_variance(target: float) -> None:
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target_variance must be in (0, 1], got {target!r}")


def select_n_components(variance_ratio: np.ndarray, target: float) -> int:
    """First k with cumulative variance >= target (scree-rule helper).

    Explicit accumulate-until-threshold loop: walks PCs in order, adding
    each one's variance, and stops at the first k reaching the target.
    Falls through to all components if rounding leaves the total just
    under the target (e.g. target=1.0 with ratios summing to 0.9999).
    """
    _validate_target_variance(target)
    cumulative = 0.0
    for k, ratio in enumerate(variance_ratio, start=1):
        cumulative += float(ratio)
        if cumulative >= target:
            return k
    return len(variance_ratio)


class AddWindCircular(BaseEstimator, TransformerMixin):
    """Replace circular wind direction with its sin/cos expansion.

    Input degrees (any range); output ``wd_sin``/``wd_cos`` in fixed
    feature order, raw direction dropped. Unweighted on purpose: wind
    speed already enters as its own feature, and direction is
    meaningless on calm winds.
    """

    def fit(self, X: pd.DataFrame, y: None = None) -> AddWindCircular:
        if WIND_DIRECTION_COL not in X.columns:
            raise KeyError(f"Missing column: {WIND_DIRECTION_COL}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        out = X.copy()
        radians = np.radians(out[WIND_DIRECTION_COL].to_numpy(dtype=float))
        out[WIND_SIN_COL] = np.sin(radians)
        out[WIND_COS_COL] = np.cos(radians)
        # Drop the raw circular column; every other column passes through
        # untouched. Column order is fixed upstream (fit_weather_pca selects
        # pca_input_columns()), so the output order is deterministic:
        # the 12 base columns followed by wd_sin/wd_cos.
        return out.drop(columns=[WIND_DIRECTION_COL])


class LogPrecipitation(BaseEstimator, TransformerMixin):
    """Apply ``log1p`` to ``precipitation`` only (highly skewed, ~95% zeros)."""

    def fit(self, X: pd.DataFrame, y: None = None) -> LogPrecipitation:
        if PRECIPITATION_COL not in X.columns:
            raise KeyError(f"Missing column: {PRECIPITATION_COL}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        out = X.copy()
        out[PRECIPITATION_COL] = np.log1p(out[PRECIPITATION_COL].clip(lower=0))
        return out


def build_weather_pca_pipeline() -> Pipeline:
    """Full-fit PCA pipeline (14 features; k selected afterwards)."""
    return Pipeline(
        [
            ("wind_circular", AddWindCircular()),
            ("log_precip", LogPrecipitation()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA()),
        ]
    )


@dataclass
class WeatherPCAResult:
    """Fitted pipeline plus the artifacts the notebook plots/uses."""

    pipeline: Pipeline
    feature_names: list[str]
    variance_ratio: np.ndarray
    n_components: int
    scores: pd.DataFrame  # first k PCs (PC1..PCk), same index as input
    loadings: pd.DataFrame  # 14 features x all PCs (eigenvectors scaled)


def fit_weather_pca(
    df: pd.DataFrame, target_variance: float | None = None
) -> WeatherPCAResult:
    """Fit the weather PCA and select k via the target variance.

    Parameters
    ----------
    df:
        DataFrame with at least :func:`pca_input_columns`.
    target_variance:
        Cumulative-variance fraction; defaults to
        ``pca.target_variance`` from config.

    Returns
    -------
    WeatherPCAResult
        Pipeline (refittable on train only for later modeling), the
        first-k PC scores, full loadings, and per-PC variance ratios
        (scree input).
    """
    target = (
        load_target_variance() if target_variance is None else float(target_variance)
    )
    _validate_target_variance(target)

    missing = set(pca_input_columns()) - set(df.columns)
    if missing:
        raise KeyError(f"Missing PCA input columns: {sorted(missing)}")
    frame = df[pca_input_columns()]
    empty = [c for c in frame.columns if frame[c].isna().all()]
    if empty:
        raise ValueError(f"PCA input columns all-NaN: {empty}")

    pipeline = build_weather_pca_pipeline()
    all_scores = pipeline.fit_transform(frame)
    pca: PCA = pipeline.named_steps["pca"]
    ratios = np.asarray(pca.explained_variance_ratio_)
    k = select_n_components(ratios, target)

    pc_all = [f"PC{i + 1}" for i in range(all_scores.shape[1])]
    pc_k = pc_all[:k]
    scores = pd.DataFrame(all_scores[:, :k], index=df.index, columns=pc_k)
    # Loadings = eigenvectors scaled by sqrt(eigenvalue).
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=pca_feature_columns(),
        columns=pc_all,
    )
    return WeatherPCAResult(
        pipeline=pipeline,
        feature_names=pca_feature_columns(),
        variance_ratio=ratios,
        n_components=k,
        scores=scores,
        loadings=loadings,
    )
