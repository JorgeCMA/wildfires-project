"""Wide-margin exceedance calibration (Plan D, route C).

Pure units (margin labeling, isotonic, prior correction, reliability/metrics,
event dedup, cross-fit) plus a small end-to-end synthetic frame. The
real-data temporal smoke test lives in the notebook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.analysis.calibration import (
    EXCEEDANCE_MARGINS_KM,
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


def _points():
    return pd.DataFrame(
        {
            "latitude": [40.0, 40.0, 41.0, 40.0],
            "longitude": [0.0, 0.001, 0.0, 0.0],
            "acq_date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-02"],
        }
    )


def _detections():
    return pd.DataFrame(
        {
            "latitude": [40.0, 40.0],
            "longitude": [0.0, 0.0],
            "acq_date": ["2023-01-01", "2023-01-02"],
        }
    )


class TestMarginPositiveLabels:
    def test_same_day_near_is_positive(self):
        hits = margin_positive_labels(_points(), _detections(), 5.0)
        # (40,0) and (40,0.001) on day 1 are within 5 km; (41,0) is not.
        assert hits.tolist() == [True, True, False, True]

    def test_small_margin_excludes_nearby(self):
        hits = margin_positive_labels(_points(), _detections(), 0.5)
        # (40,0.001) ~ 85 m < 0.5 km still true; only the exact point in day 2
        assert hits.tolist() == [True, True, False, True]

    def test_zero_margin_only_exact_location(self):
        points = pd.DataFrame(
            {
                "latitude": [40.0, 40.001],
                "longitude": [0.0, 0.0],
                "acq_date": ["2023-01-01", "2023-01-01"],
            }
        )
        hits = margin_positive_labels(points, _detections(), 0.0)
        assert hits.tolist() == [True, False]

    def test_wrong_day_is_negative(self):
        detection = pd.DataFrame(
            {"latitude": [40.0], "longitude": [0.0], "acq_date": ["2023-01-05"]}
        )
        hits = margin_positive_labels(_points(), detection, 5.0)
        assert not hits.any()

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin_km"):
            margin_positive_labels(_points(), _detections(), -1.0)

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="latitude"):
            margin_positive_labels(
                _points().drop(columns=["latitude"]), _detections(), 5.0
            )


class TestIsotonic:
    def test_fit_apply_monotone(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
        model = fit_isotonic(scores, y)
        predicted = apply_isotonic(model, scores)
        assert (np.diff(predicted) >= -1e-12).all()
        assert predicted.min() >= 0.0 and predicted.max() <= 1.0

    def test_needs_both_classes(self):
        with pytest.raises(ValueError, match="both classes"):
            fit_isotonic(np.array([0.1, 0.2, 0.3]), np.array([1, 1, 1]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            fit_isotonic(np.array([0.1, 0.2]), np.array([0, 1, 0]))

    def test_nan_scores_raise(self):
        with pytest.raises(ValueError, match="finite"):
            apply_isotonic(fit_isotonic([0.1, 0.5, 0.9], [0, 1, 1]), [np.nan])


class TestPriorCorrect:
    def test_identity_when_prevalences_equal(self):
        p = np.array([0.1, 0.4, 0.8])
        out = prior_correct(p, source_prevalence=0.3, target_prevalence=0.3)
        assert np.allclose(out, p, atol=1e-9)

    def test_target_below_source_shrinks(self):
        p = np.array([0.5, 0.7])
        out = prior_correct(p, source_prevalence=0.5, target_prevalence=0.05)
        assert (out < p).all()

    def test_target_above_source_grows(self):
        out = prior_correct(
            np.array([0.2, 0.4]), source_prevalence=0.1, target_prevalence=0.3
        )
        assert (out > np.array([0.2, 0.4])).all()

    def test_bad_prevalence_raises(self):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            prior_correct([0.5], source_prevalence=0.0, target_prevalence=0.5)
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            prior_correct([0.5], source_prevalence=0.5, target_prevalence=1.0)


class TestReliabilityAndMetrics:
    def test_perfect_predictor_is_calibrated(self):
        y = np.array([0] * 50 + [1] * 50)
        p = y.astype(float)
        curve = reliability_curve(y, p, n_bins=5)
        assert curve["n"].sum() == 100
        assert np.allclose(curve["mean_predicted"], curve["observed_freq"], atol=0.05)

    def test_metrics_perfect_are_near_zero(self):
        y = np.array([0] * 40 + [1] * 40)
        p = y.astype(float)
        metrics = calibration_metrics(y, p)
        assert metrics["brier"] < 0.01
        assert metrics["ece"] < 0.01

    def test_quantile_bins_survive_duplicate_scores(self):
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        p = np.array([0.5] * 8)
        curve = reliability_curve(y, p, n_bins=4, strategy="quantile")
        assert curve["n"].sum() == 8

    def test_uniform_edges(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.05, 0.95, 0.15, 0.85])
        curve = reliability_curve(y, p, n_bins=2, strategy="uniform")
        assert len(curve) == 2

    def test_bad_strategy_raises(self):
        with pytest.raises(ValueError, match="strategy"):
            reliability_curve([0, 1], [0.2, 0.8], strategy="nope")

    def test_metrics_need_both_classes(self):
        with pytest.raises(ValueError, match="both classes"):
            calibration_metrics([1, 1, 1], [0.6, 0.7, 0.8])


class TestDedupeEvents:
    def test_one_row_per_event(self):
        detections = pd.DataFrame({"detection_id": [1, 2, 3]})
        events = pd.DataFrame(
            {"detection_id": [1, 2, 3], "event_id": ["e1", "e1", "e2"]}
        )
        out = dedupe_events(detections, events)
        assert out["detection_id"].tolist() == [1, 3]
        assert out["event_id"].tolist() == ["e1", "e2"]

    def test_missing_event_col_raises(self):
        with pytest.raises(KeyError, match="event_id"):
            dedupe_events(
                pd.DataFrame({"detection_id": [1]}),
                pd.DataFrame({"detection_id": [1]}),
            )


class TestExceedanceBaseRate:
    def test_half_near_detection(self):
        background = pd.DataFrame(
            {
                "latitude": [40.0, 40.001, 41.0, 41.0],
                "longitude": [0.0, 0.001, 0.0, 0.0],
                "acq_date": ["2023-01-01", "2023-01-01", "2023-01-01", "2023-01-01"],
            }
        )
        result = exceedance_base_rate(background, _detections(), 5.0, n_boot=200)
        assert result["n_points"] == 4
        assert result["n_positive"] == 2
        assert result["rate"] == pytest.approx(0.5)
        assert result["ci_low"] <= result["rate"] <= result["ci_high"]


class TestCrossFittedIsotonic:
    def test_pooled_predictions_complete_and_bounded(self):
        scores = np.linspace(0.05, 0.95, 40)
        y = (scores > 0.5).astype(int)
        days = np.repeat(["d1", "d2", "d3", "d4"], 10)
        pooled, final = cross_fitted_isotonic(scores, y, days, n_splits=4)
        assert not np.isnan(pooled).any()
        assert pooled.min() >= 0.0 and pooled.max() <= 1.0
        assert final.predict([0.6])[0] > final.predict([0.4])[0]

    def test_too_few_splits_raises(self):
        with pytest.raises(ValueError, match="n_splits"):
            cross_fitted_isotonic([0.1, 0.9], [0, 1], ["d1", "d1"], n_splits=1)

    def test_needs_both_classes(self):
        with pytest.raises(ValueError, match="both classes"):
            cross_fitted_isotonic([0.1, 0.9], [1, 1], ["d1", "d2"])


class TestCalibrateExceedance:
    def _frames(self):
        validation = pd.DataFrame(
            {
                "score": [0.9, 0.8, 0.2, 0.85, 0.15],
                "latitude": [40.0, 40.0, 41.0, 40.0, 41.0],
                "longitude": [0.001, 0.002, 0.0, 0.001, 0.0],
                "acq_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ],
            }
        )
        background = pd.DataFrame(
            {
                "latitude": [40.0, 40.001, 41.0, 41.0],
                "longitude": [0.0, 0.001, 0.0, 0.0],
                "acq_date": [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                ],
            }
        )
        detections = pd.DataFrame(
            {
                "latitude": [40.0, 40.0],
                "longitude": [0.0, 0.0],
                "acq_date": ["2024-01-01", "2024-01-02"],
            }
        )
        return validation, background, detections

    def test_returns_one_result_per_margin(self):
        validation, background, detections = self._frames()
        results = calibrate_exceedance(
            validation,
            detections,
            background,
            margins=(5.0, 10.0),
            n_splits=2,
            n_bins=3,
            n_boot=100,
        )
        assert set(results) == {5.0, 10.0}
        for result in results.values():
            assert 0.0 <= result.metrics["brier"] <= 1.0
            assert result.n_points == len(validation)
            assert result.reliability["n"].sum() == len(validation)

    def test_summary_columns(self):
        validation, background, detections = self._frames()
        results = calibrate_exceedance(
            validation,
            detections,
            background,
            margins=(5.0,),
            n_splits=2,
            n_bins=3,
            n_boot=100,
        )
        summary = exceedance_summary(results)
        assert list(summary.columns) == [
            "margin_km",
            "n_points",
            "n_margin_positive",
            "source_prevalence",
            "base_rate",
            "prior_corrected_mean",
            "brier",
            "log_loss",
            "ece",
        ]
        assert summary.iloc[0]["margin_km"] == 5.0

    def test_single_class_margin_raises(self):
        validation, background, _ = self._frames()
        # A far-away detection makes every validation point a true negative.
        far_detection = pd.DataFrame(
            {"latitude": [10.0], "longitude": [0.0], "acq_date": ["2024-01-01"]}
        )
        with pytest.raises(ValueError, match="single class"):
            calibrate_exceedance(
                validation,
                far_detection,
                background,
                margins=(1.0,),
                n_splits=2,
                n_bins=3,
                n_boot=100,
            )

    def test_missing_score_raises(self):
        validation, background, _ = self._frames()
        with pytest.raises(KeyError, match="score"):
            calibrate_exceedance(
                validation.drop(columns=["score"]),
                _detections(),
                background,
                margins=(5.0,),
            )


def test_default_margins_constant():
    assert EXCEEDANCE_MARGINS_KM == (5.0, 10.0, 20.0)
