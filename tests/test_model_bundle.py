"""Tests for the persisted occurrence bundle (preprocessing + calibration)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from wildfire.analysis.calibration import apply_isotonic, fit_isotonic
from wildfire.analysis.model_bundle import (
    MarginCalibrator,
    OccurrenceBundle,
    calibrated_exceedance,
    fit_preprocessing,
    load_bundle,
    predict_score,
    save_bundle,
    score_point,
    transform,
)

# Real everyday column names: ``continuous_columns`` only keeps columns from
# the fixed EVERYDAY_FEATURES contract, so synthetic names would be ignored.
FEATURE_COLS = ["latitude", "temperature_2m", "ccaa"]
CATEGORICAL = ["ccaa"]


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": np.arange(n, dtype=float) + 35.0,
            "temperature_2m": np.linspace(10.0, 30.0, n),
            "ccaa": ["Madrid", "Galicia"] * (n // 2),
            "label": [0, 1] * (n // 2),
        }
    )


class TestFitPreprocessingAndTransform:
    def test_round_trip_shape_and_order(self):
        train = _frame(10)
        prep = fit_preprocessing(train, FEATURE_COLS, CATEGORICAL)
        matrix = transform(train, prep)

        assert len(prep.continuous) == 2
        assert prep.dummies == ["ccaa_Galicia", "ccaa_Madrid"]
        assert list(matrix.columns) == prep.feature_order
        assert matrix.shape == (10, 4)

    def test_continuous_columns_are_standardized(self):
        prep = fit_preprocessing(_frame(10), FEATURE_COLS, CATEGORICAL)
        matrix = transform(_frame(10), prep)

        assert matrix["temperature_2m"].mean() == pytest.approx(0.0, abs=1e-9)
        assert matrix["temperature_2m"].std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    def test_unseen_category_becomes_all_zero(self):
        prep = fit_preprocessing(_frame(10), FEATURE_COLS, CATEGORICAL)
        new = pd.DataFrame(
            {"latitude": [38.0], "temperature_2m": [20.0], "ccaa": ["Wonderland"]}
        )

        matrix = transform(new, prep)

        assert matrix["ccaa_Galicia"].iloc[0] == 0.0
        assert matrix["ccaa_Madrid"].iloc[0] == 0.0

    def test_missing_continuous_column_uses_median(self):
        prep = fit_preprocessing(_frame(10), FEATURE_COLS, CATEGORICAL)
        new = pd.DataFrame({"latitude": [38.0], "ccaa": ["Madrid"]})  # no weather

        matrix = transform(new, prep)

        assert not matrix.isna().any().any()
        index = prep.continuous.index("temperature_2m")
        expected = (prep.medians["temperature_2m"] - prep.scaler.mean_[index]) / (
            prep.scaler.scale_[index]
        )
        assert matrix["temperature_2m"].iloc[0] == pytest.approx(expected)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        train = _frame(12)
        prep = fit_preprocessing(train, FEATURE_COLS, CATEGORICAL)
        model = LogisticRegression(max_iter=1000).fit(
            transform(train, prep), train["label"]
        )
        bundle = OccurrenceBundle(model=model, preprocessing=prep, margins={})
        path = save_bundle(bundle, tmp_path / "nested" / "bundle.joblib")

        assert path.exists()
        restored = load_bundle(path)
        assert isinstance(restored, OccurrenceBundle)
        assert restored.preprocessing.feature_order == prep.feature_order
        assert restored.preprocessing.dummies == prep.dummies

        one = pd.DataFrame(
            {"latitude": [38.0], "temperature_2m": [22.0], "ccaa": ["Madrid"]}
        )
        assert predict_score(restored, one) == pytest.approx(predict_score(bundle, one))

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_bundle(tmp_path / "absent.joblib")


class TestCalibratedExceedance:
    def _bundle(self) -> OccurrenceBundle:
        calibrator = fit_isotonic(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1])
        )
        return OccurrenceBundle(
            model=None,
            preprocessing=fit_preprocessing(_frame(12), FEATURE_COLS, CATEGORICAL),
            margins={
                5.0: MarginCalibrator(5.0, calibrator, 0.5, 0.5),
                10.0: MarginCalibrator(10.0, calibrator, 0.5, 0.05),
            },
        )

    def test_equal_prevalence_is_identity_correction(self):
        bundle = self._bundle()
        expected = apply_isotonic(bundle.margins[5.0].calibrator, np.array([2.0]))[0]

        # prior_correct clips off 0/1, so equality is up to the clipping epsilon.
        assert calibrated_exceedance(bundle, 2.0)[5.0] == pytest.approx(
            expected, abs=1e-4
        )

    def test_target_below_source_shrinks(self):
        result = calibrated_exceedance(self._bundle(), 2.0)

        assert result[10.0] < result[5.0]


class TestScorePoint:
    def test_returns_score_and_per_margin_exceedance(self):
        train = _frame(12)
        prep = fit_preprocessing(train, FEATURE_COLS, CATEGORICAL)
        model = LogisticRegression(max_iter=1000).fit(
            transform(train, prep), train["label"]
        )
        calibrator = fit_isotonic(np.linspace(0, 1, 12), train["label"])
        bundle = OccurrenceBundle(
            model=model,
            preprocessing=prep,
            margins={5.0: MarginCalibrator(5.0, calibrator, 0.4, 0.1)},
        )
        one = pd.DataFrame(
            {"latitude": [38.0], "temperature_2m": [22.0], "ccaa": ["Madrid"]}
        )

        result = score_point(bundle, one)

        assert 0.0 <= result["score"] <= 1.0
        assert set(result["exceedance"]) == {5.0}
        assert 0.0 <= result["exceedance"][5.0] <= 1.0
