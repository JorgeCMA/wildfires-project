"""Tests for confidence mapping between MODIS numerical and VIIRS categorical values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.confidence import (
    _load_thresholds,
    add_unified_confidence,
    categorical_to_numerical,
    numerical_to_categorical,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def thresholds() -> dict:
    """Load thresholds from configs/confidence_thresholds.yaml."""
    return _load_thresholds()


# ---------------------------------------------------------------------------
# _load_thresholds
# ---------------------------------------------------------------------------

class TestLoadThresholds:
    def test_returns_dict(self):
        result = _load_thresholds()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = _load_thresholds()
        assert "numerical_to_categorical" in result
        assert "categorical_to_numerical" in result

    def test_numerical_ranges_cover_0_to_100(self):
        result = _load_thresholds()
        ranges = result["numerical_to_categorical"]
        assert ranges["low"]["min"] == 0
        assert ranges["high"]["max"] == 100


# ---------------------------------------------------------------------------
# numerical_to_categorical
# ---------------------------------------------------------------------------

class TestNumericalToCategorical:
    def test_low_range(self, thresholds):
        assert numerical_to_categorical(0, thresholds) == "l"
        assert numerical_to_categorical(15, thresholds) == "l"
        assert numerical_to_categorical(29, thresholds) == "l"
        assert numerical_to_categorical(30, thresholds) == "l"

    def test_nominal_range(self, thresholds):
        assert numerical_to_categorical(31, thresholds) == "n"
        assert numerical_to_categorical(50, thresholds) == "n"
        assert numerical_to_categorical(69, thresholds) == "n"
        assert numerical_to_categorical(70, thresholds) == "n"

    def test_high_range(self, thresholds):
        assert numerical_to_categorical(71, thresholds) == "h"
        assert numerical_to_categorical(85, thresholds) == "h"
        assert numerical_to_categorical(99, thresholds) == "h"
        assert numerical_to_categorical(100, thresholds) == "h"

    def test_boundary_overlaps_are_inclusive(self, thresholds):
        # 30 is in both low and nominal ranges (both use <=)
        # First match wins: low range (0 <= 30 <= 30)
        assert numerical_to_categorical(30, thresholds) == "l"
        # 70 is in both nominal and high ranges (both use <=)
        # First match wins: nominal range (30 <= 70 <= 70)
        assert numerical_to_categorical(70, thresholds) == "n"

    def test_nan_returns_n(self, thresholds):
        assert numerical_to_categorical(float("nan"), thresholds) == "n"
        assert numerical_to_categorical(np.nan, thresholds) == "n"
        assert numerical_to_categorical(pd.NA, thresholds) == "n"

    def test_negative_raises(self, thresholds):
        with pytest.raises(ValueError, match="outside valid range"):
            numerical_to_categorical(-1, thresholds)

    def test_above_100_raises(self, thresholds):
        with pytest.raises(ValueError, match="outside valid range"):
            numerical_to_categorical(101, thresholds)

    def test_far_out_of_range_raises(self, thresholds):
        with pytest.raises(ValueError):
            numerical_to_categorical(-999, thresholds)
        with pytest.raises(ValueError):
            numerical_to_categorical(999, thresholds)

    def test_return_type_is_str(self, thresholds):
        result = numerical_to_categorical(50, thresholds)
        assert isinstance(result, str)

    def test_all_returns_are_single_char(self, thresholds):
        for val in [0, 15, 30, 31, 50, 69, 70, 71, 85, 100]:
            result = numerical_to_categorical(val, thresholds)
            assert len(result) == 1, f"value={val} returned {result!r}"


# ---------------------------------------------------------------------------
# categorical_to_numerical
# ---------------------------------------------------------------------------

class TestCategoricalToNumerical:
    def test_low(self, thresholds):
        assert categorical_to_numerical("l", thresholds) == 15

    def test_nominal(self, thresholds):
        assert categorical_to_numerical("n", thresholds) == 50

    def test_high(self, thresholds):
        assert categorical_to_numerical("h", thresholds) == 85

    def test_case_insensitive(self, thresholds):
        assert categorical_to_numerical("L", thresholds) == 15
        assert categorical_to_numerical("N", thresholds) == 50
        assert categorical_to_numerical("H", thresholds) == 85

    def test_whitespace_stripped(self, thresholds):
        assert categorical_to_numerical("  l  ", thresholds) == 15
        assert categorical_to_numerical(" n ", thresholds) == 50

    def test_unknown_defaults_to_nominal(self, thresholds):
        assert categorical_to_numerical("x", thresholds) == 50
        assert categorical_to_numerical("unknown", thresholds) == 50
        assert categorical_to_numerical("", thresholds) == 50

    def test_nan_returns_nominal_value(self, thresholds):
        assert categorical_to_numerical(float("nan"), thresholds) == 50
        assert categorical_to_numerical(np.nan, thresholds) == 50
        assert categorical_to_numerical(pd.NA, thresholds) == 50

    def test_return_type_is_float(self, thresholds):
        result = categorical_to_numerical("l", thresholds)
        assert isinstance(result, (int, float))


# ---------------------------------------------------------------------------
# add_unified_confidence
# ---------------------------------------------------------------------------

def _make_firms_df(sensors: list[str], confidences: list[str]) -> pd.DataFrame:
    """Build a minimal FIRMS DataFrame for testing."""
    n = len(sensors)
    return pd.DataFrame({
        "latitude": [40.0] * n,
        "longitude": [-3.0] * n,
        "acq_date": ["2023-07-01"] * n,
        "acq_time": ["1200"] * n,
        "sensor": sensors,
        "confidence": confidences,
        "brightness": [300.0] * n,
        "frp": [10.0] * n,
    })


class TestAddUnifiedConfidence:
    def test_output_columns_present(self):
        df = _make_firms_df(
            ["modis", "viirs_snpp", "viirs_noaa20"],
            ["50", "n", "h"],
        )
        result = add_unified_confidence(df)
        for col in ["confidence_og_num", "confidence_og_cat", "confidence_cat", "confidence_num"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self):
        df = _make_firms_df(["modis"], ["50"])
        original_cols = list(df.columns)
        add_unified_confidence(df)
        assert list(df.columns) == original_cols

    def test_modis_og_num_populated(self):
        df = _make_firms_df(["modis", "modis"], ["45", "80"])
        result = add_unified_confidence(df)
        assert str(result.loc[0, "confidence_og_num"]) == "45"
        assert str(result.loc[1, "confidence_og_num"]) == "80"

    def test_modis_og_cat_is_na(self):
        df = _make_firms_df(["modis"], ["50"])
        result = add_unified_confidence(df)
        assert pd.isna(result.loc[0, "confidence_og_cat"])

    def test_viirs_og_cat_populated(self):
        df = _make_firms_df(["viirs_snpp", "viirs_noaa20"], ["l", "h"])
        result = add_unified_confidence(df)
        assert result.loc[0, "confidence_og_cat"] == "l"
        assert result.loc[1, "confidence_og_cat"] == "h"

    def test_viirs_og_num_is_na(self):
        df = _make_firms_df(["viirs_snpp"], ["n"])
        result = add_unified_confidence(df)
        assert pd.isna(result.loc[0, "confidence_og_num"])

    def test_modis_categorical_conversion(self):
        df = _make_firms_df(["modis", "modis", "modis"], ["10", "50", "90"])
        result = add_unified_confidence(df)
        assert result.loc[0, "confidence_cat"] == "l"
        assert result.loc[1, "confidence_cat"] == "n"
        assert result.loc[2, "confidence_cat"] == "h"

    def test_modis_numerical_preserved(self):
        df = _make_firms_df(["modis"], ["50"])
        result = add_unified_confidence(df)
        assert result.loc[0, "confidence_num"] == 50

    def test_viirs_numerical_conversion(self):
        df = _make_firms_df(["viirs_snpp", "viirs_snpp", "viirs_snpp"], ["l", "n", "h"])
        result = add_unified_confidence(df)
        assert result.loc[0, "confidence_num"] == 15
        assert result.loc[1, "confidence_num"] == 50
        assert result.loc[2, "confidence_num"] == 85

    def test_viirs_categorical_preserved(self):
        df = _make_firms_df(["viirs_noaa20"], ["h"])
        result = add_unified_confidence(df)
        assert result.loc[0, "confidence_cat"] == "h"

    def test_mixed_sensors_all_rows_filled(self):
        df = _make_firms_df(
            ["modis", "viirs_snpp", "viirs_noaa20", "modis"],
            ["60", "l", "h", "20"],
        )
        result = add_unified_confidence(df)
        assert result["confidence_cat"].notna().all()
        assert result["confidence_num"].notna().all()

    def test_confidence_cat_is_category_dtype(self):
        df = _make_firms_df(["modis", "viirs_snpp"], ["50", "n"])
        result = add_unified_confidence(df)
        assert result["confidence_cat"].dtype.name == "category"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time", "sensor", "confidence", "brightness", "frp"])
        result = add_unified_confidence(df)
        assert "confidence_cat" in result.columns
        assert "confidence_num" in result.columns
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Integration tests — run against actual enriched dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def enriched_df() -> pd.DataFrame:
    """Load the real enriched FIRMS dataset."""
    path = r"C:\Projects\wildfires-project\data\processed\enriched\firms_spain_enriched.csv"
    return pd.read_csv(path)


class TestAgainstRealData:
    def test_enriched_has_confidence_columns(self, enriched_df):
        for col in ["confidence_cat", "confidence_num", "confidence_og_num", "confidence_og_cat"]:
            assert col in enriched_df.columns

    def test_no_missing_unified_confidence(self, enriched_df):
        assert enriched_df["confidence_cat"].notna().all()
        assert enriched_df["confidence_num"].notna().all()

    def test_all_sensors_present(self, enriched_df):
        sensors = set(enriched_df["sensor"].unique())
        assert "modis" in sensors
        assert "viirs_snpp" in sensors or "viirs_noaa20" in sensors

    def test_modis_rows_have_numerical_original(self, enriched_df):
        modis = enriched_df[enriched_df["sensor"] == "modis"]
        assert modis["confidence_og_num"].notna().all()

    def test_viirs_rows_have_categorical_original(self, enriched_df):
        viirs = enriched_df[enriched_df["sensor"].str.startswith("viirs")]
        assert viirs["confidence_og_cat"].notna().all()

    def test_modis_categorical_values_are_valid(self, enriched_df):
        modis = enriched_df[enriched_df["sensor"] == "modis"]
        valid = {"l", "n", "h"}
        assert set(modis["confidence_cat"].unique()).issubset(valid)

    def test_viirs_numerical_values_in_range(self, enriched_df):
        viirs = enriched_df[enriched_df["sensor"].str.startswith("viirs")]
        assert (viirs["confidence_num"] >= 0).all()
        assert (viirs["confidence_num"] <= 100).all()


# ---------------------------------------------------------------------------
# Edge-case tests — artificial data designed to expose failures
# ---------------------------------------------------------------------------

class TestEdgeCasesThatShouldFail:
    """Tests that deliberately use bad data to verify the code rejects it."""

    def test_modis_value_exactly_100_works(self, thresholds):
        assert numerical_to_categorical(100, thresholds) == "h"

    def test_modis_value_exactly_0_works(self, thresholds):
        assert numerical_to_categorical(0, thresholds) == "l"

    def test_modis_value_100_point_1_raises(self, thresholds):
        with pytest.raises(ValueError):
            numerical_to_categorical(100.1, thresholds)

    def test_viirs_confidence_with_extra_whitespace(self, thresholds):
        assert categorical_to_numerical("  H  ", thresholds) == 85

    def test_mixed_modis_viirs_all_confidence_filled(self):
        df = _make_firms_df(
            ["modis"] * 3 + ["viirs_snpp"] * 3,
            ["0", "50", "100", "l", "n", "h"],
        )
        result = add_unified_confidence(df)
        # Every row must have a unified confidence
        assert result["confidence_cat"].notna().all()
        assert result["confidence_num"].notna().all()

    def test_modis_extreme_low_confidence(self, thresholds):
        assert numerical_to_categorical(0, thresholds) == "l"

    def test_modis_extreme_high_confidence(self, thresholds):
        assert numerical_to_categorical(100, thresholds) == "h"

    def test_viirs_all_categories_map_correctly(self, thresholds):
        expected = {"l": 15, "n": 50, "h": 85}
        for cat, num in expected.items():
            assert categorical_to_numerical(cat, thresholds) == num

    def test_add_unified_preserves_sensor_column(self):
        df = _make_firms_df(["modis", "viirs_snpp"], ["50", "n"])
        result = add_unified_confidence(df)
        assert list(result["sensor"]) == ["modis", "viirs_snpp"]

    def test_add_unified_does_not_drop_rows(self):
        df = _make_firms_df(["modis", "viirs_noaa20", "modis"], ["30", "l", "90"])
        result = add_unified_confidence(df)
        assert len(result) == 3

    def test_boundary_value_30_is_low_not_nominal(self, thresholds):
        assert numerical_to_categorical(30, thresholds) == "l"

    def test_boundary_value_70_is_nominal_not_high(self, thresholds):
        assert numerical_to_categorical(70, thresholds) == "n"

    def test_non_numeric_modis_confidence_coerced_to_nan(self):
        df = _make_firms_df(["modis"], ["not_a_number"])
        result = add_unified_confidence(df)
        # pd.to_numeric with errors="coerce" turns this to NaN
        assert pd.isna(result.loc[0, "confidence_num"])
        # NaN falls back to "n"
        assert result.loc[0, "confidence_cat"] == "n"
