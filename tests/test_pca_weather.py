import numpy as np
import pandas as pd
import pytest

from wildfire.processing.pca_weather import (
    WIND_COS_COL,
    WIND_SIN_COL,
    AddWindCircular,
    LogPrecipitation,
    build_weather_pca_pipeline,
    fit_weather_pca,
    load_target_variance,
    pca_feature_columns,
    pca_input_columns,
    select_n_components,
)
from wildfire.processing.typology import K12_PC_COLUMNS


def _make_weather_df(n=60, seed=7):
    rng = np.random.default_rng(seed)
    data = {
        "temperature_2m": rng.normal(15, 6, n),
        "relative_humidity_2m": rng.uniform(20, 95, n),
        "vapour_pressure_deficit": rng.uniform(0, 3, n),
        "et0_fao_evapotranspiration": rng.uniform(0, 0.6, n),
        "shortwave_radiation": rng.uniform(0, 800, n),
        "boundary_layer_height": rng.uniform(50, 1500, n),
        "surface_pressure": rng.normal(1000, 20, n),
        "wind_speed_10m": rng.uniform(0, 25, n),
        "wind_gusts_10m": rng.uniform(0, 45, n),
        "wind_direction_10m": rng.uniform(0, 360, n),
        "precipitation": rng.exponential(0.2, n),
        "soil_moisture_0_to_7cm": rng.uniform(0, 0.45, n),
        "soil_moisture_7_to_28cm": rng.uniform(0, 0.45, n),
    }
    return pd.DataFrame(data)


class TestFeatureColumns:
    def test_fourteen_features(self):
        assert len(pca_feature_columns()) == 14

    def test_raw_direction_excluded_sincos_included(self):
        assert "wind_direction_10m" not in pca_feature_columns()
        assert WIND_SIN_COL in pca_feature_columns()
        assert WIND_COS_COL in pca_feature_columns()

    def test_rain_and_is_day_excluded(self):
        assert "rain" not in pca_feature_columns()
        assert "is_day" not in pca_feature_columns()


class TestAddWindCircular:
    def test_cardinal_angles(self):
        df = pd.DataFrame({"wind_direction_10m": [0.0, 90.0, 180.0, 270.0]})
        out = AddWindCircular().fit_transform(df)
        assert np.allclose(out[WIND_SIN_COL], [0, 1, 0, -1], atol=1e-9)
        assert np.allclose(out[WIND_COS_COL], [1, 0, -1, 0], atol=1e-9)

    def test_360_equals_0(self):
        df = pd.DataFrame({"wind_direction_10m": [360.0]})
        out = AddWindCircular().fit_transform(df)
        assert np.allclose(out[WIND_SIN_COL], [0.0], atol=1e-9)
        assert np.allclose(out[WIND_COS_COL], [1.0], atol=1e-9)

    def test_raw_column_dropped(self):
        df = pd.DataFrame({c: [1.0] for c in pca_input_columns()})
        out = AddWindCircular().fit_transform(df)
        assert "wind_direction_10m" not in out.columns
        assert list(out.columns) == pca_feature_columns()

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            AddWindCircular().fit(pd.DataFrame({"a": [1.0]}))


class TestLogPrecipitation:
    def test_log1p_applied_only_to_precipitation(self):
        df = pd.DataFrame({c: [2.0] for c in pca_feature_columns()})
        out = LogPrecipitation().fit_transform(df)
        assert out["precipitation"].iloc[0] == pytest.approx(np.log1p(2.0))
        assert out["temperature_2m"].iloc[0] == pytest.approx(2.0)


class TestSelectNComponents:
    def test_first_k_reaching_target(self):
        ratios = np.array([0.5, 0.25, 0.15, 0.10])
        assert select_n_components(ratios, 0.85) == 3
        assert select_n_components(ratios, 0.5) == 1

    def test_rounding_fallthrough_returns_all(self):
        ratios = np.array([0.5, 0.4999999])
        assert select_n_components(ratios, 1.0) == 2

    def test_invalid_target_raises(self):
        with pytest.raises(ValueError):
            select_n_components(np.array([0.5, 0.5]), 0.0)
        with pytest.raises(ValueError):
            select_n_components(np.array([0.5, 0.5]), 1.5)


class TestLoadTargetVariance:
    def test_config_default(self):
        assert load_target_variance() == pytest.approx(0.90)


class TestFitWeatherPca:
    def test_end_to_end_shapes(self):
        df = _make_weather_df()
        result = fit_weather_pca(df)
        assert result.n_components >= 1
        assert result.scores.shape == (len(df), result.n_components)
        assert result.loadings.shape == (14, 14)
        assert len(result.variance_ratio) == 14
        assert result.scores.columns.tolist()[0] == "PC1"

    def test_scores_finite_and_centered(self):
        df = _make_weather_df()
        result = fit_weather_pca(df)
        assert np.isfinite(result.scores.to_numpy()).all()
        assert abs(result.scores["PC1"].mean()) < 1e-9

    def test_k_matches_target(self):
        df = _make_weather_df()
        result = fit_weather_pca(df, target_variance=0.85)
        assert result.variance_ratio[: result.n_components].sum() >= 0.85

    def test_missing_column_raises(self):
        df = _make_weather_df().drop(columns=["temperature_2m"])
        with pytest.raises(KeyError):
            fit_weather_pca(df)

    def test_all_nan_column_raises(self):
        df = _make_weather_df()
        df["surface_pressure"] = np.nan
        with pytest.raises(ValueError):
            fit_weather_pca(df)

    def test_pipeline_reusable_on_new_data(self):
        df = _make_weather_df()
        result = fit_weather_pca(df)
        new = _make_weather_df(n=10, seed=99)[pca_input_columns()]
        scores = result.pipeline.transform(new)
        assert scores.shape == (10, 14)

    def test_builder_step_order(self):
        pipe = build_weather_pca_pipeline()
        assert [name for name, _ in pipe.steps] == [
            "wind_circular",
            "log_precip",
            "impute",
            "scale",
            "pca",
        ]


class TestAgainstRealFeaturesA:
    """WS8: ``fit_weather_pca`` reproduces the 03a/03b ``features_a.csv`` space.

    ``BackgroundScorer`` (reintegration) refits the weather PCA on the positives
    while the persisted k12 typology was fit on ``features_a.csv`` (03a). This
    test pins the two to the same space: PC1-7, k = 7 and cumulative variance
    ~0.915, so positive ``cluster_meteo`` labels match the persisted typology.
    """

    def test_refit_reproduces_features_a(self):
        from wildfire.data.io import load_csv

        try:
            positives = load_csv(
                "data/processed/expanded/firms_spain_final_lags_index.csv"
            )
            features_a = load_csv("data/processed/unsupervised/features_a.csv")
        except FileNotFoundError:
            pytest.skip("expanded/03 artifacts not on disk")

        result = fit_weather_pca(positives)

        assert result.n_components == len(K12_PC_COLUMNS)
        assert result.variance_ratio[: result.n_components].sum() == pytest.approx(
            0.915, abs=0.01
        )

        scores = result.scores.copy()
        scores["detection_id"] = positives["detection_id"].to_numpy()
        merged = scores.merge(features_a, on="detection_id", suffixes=("_new", "_old"))
        assert len(merged) == len(positives)

        for pc in K12_PC_COLUMNS:
            new = merged[f"{pc}_new"].to_numpy()
            old = merged[f"{pc}_old"].to_numpy()
            # PCA sign is indeterminate: match up to a global sign flip.
            diverges = min(np.max(np.abs(new - old)), np.max(np.abs(new + old)))
            assert diverges == pytest.approx(0.0, abs=1e-9), (
                f"{pc} diverges from features_a"
            )
