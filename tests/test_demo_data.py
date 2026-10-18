"""Tests for the demo's read-only loaders (``analysis/demo_data.py``).

Most of these load the real artifacts from disk (same convention as other
real-data tests in the repo); the missing-file cases use a bogus path so they
run anywhere.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wildfire.analysis import demo_data


class TestLoadDetections:
    def test_columns_and_dtype(self):
        df = demo_data.load_detections()
        assert {"latitude", "longitude", "acq_date", "ccaa"} <= set(df.columns)
        assert pd.api.types.is_datetime64_any_dtype(df["acq_date"])
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            demo_data.load_detections(path=str(tmp_path / "nope.csv"))


class TestLoadScores:
    def test_filters_protocol(self):
        scores = demo_data.load_scores("season")
        assert set(scores["protocol"]) == {"season"}
        assert scores["risk_band"].notna().all()

    def test_unknown_protocol_raises(self):
        with pytest.raises(ValueError):
            demo_data.load_scores("does-not-exist")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            demo_data.load_scores("season", path=str(tmp_path / "nope.csv"))


class TestBandCounts:
    def test_ordered_low_to_high(self):
        scores = demo_data.load_scores("season")
        counts = demo_data.band_counts(scores)
        assert list(counts.index) == ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
        assert counts.sum() == len(scores)


class TestChoroplethGeojson:
    def test_translates_names_and_joins_rate(self):
        rates = demo_data.load_ccaa_rates()
        geo = demo_data.ccaa_base_rate_geojson(rates)
        props = [feature["properties"] for feature in geo["features"]]
        names = {prop["ccaa_es"] for prop in props}
        assert "Castilla y León" in names
        assert "Andalucía" in names
        joined = {
            prop["ccaa_es"]: prop["fire_rate"]
            for prop in props
            if prop.get("fire_rate") is not None
        }
        assert joined, "at least one region should carry the joined rate"
        assert all(set(p) >= {"ccaa_es", "fire_rate"} for p in props)


class TestOtherLoaders:
    def test_rates_and_reliability_columns(self):
        rates = demo_data.load_ccaa_rates()
        assert {"ccaa", "fire_rate"} <= set(rates.columns)
        assert (rates["ccaa"] == "ALL").any()

        rel = demo_data.load_reliability()
        assert {"margin_km", "mean_predicted", "observed_freq"} <= set(rel.columns)
        assert set(rel["margin_km"]) == {5.0, 10.0, 20.0}

    def test_fold_metrics_columns(self):
        fold = demo_data.load_fold_metrics()
        assert {"protocol", "pr_auc", "roc_auc"} <= set(fold.columns)
