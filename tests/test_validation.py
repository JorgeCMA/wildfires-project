"""Tests for data quality validation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT
from wildfire.processing.validation import (
    validate_enriched,
    validate_enriched_clc,
    validate_enriched_openmeteo,
    validate_firms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_firms_df(n: int = 1, **overrides) -> pd.DataFrame:
    """Build a minimal valid FIRMS DataFrame with *n* rows.

    Scalar overrides are broadcast to all *n* rows. List overrides must
    have length ``n``.
    """
    base = {
        "latitude": [40.0] * n,
        "longitude": [-3.0] * n,
        "acq_date": ["2023-07-01"] * n,
        "acq_time": ["1200"] * n,
        "confidence": ["50"] * n,
        "brightness": [300.0] * n,
        "frp": [10.0] * n,
        "sensor": ["modis"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# validate_firms
# ---------------------------------------------------------------------------

class TestValidateFirms:
    def test_valid_df_no_warnings(self):
        df = _make_firms_df()
        assert validate_firms(df) == []

    def test_empty_df_warns(self):
        df = pd.DataFrame()
        warnings = validate_firms(df)
        assert any("empty" in w.lower() for w in warnings)

    def test_missing_required_columns(self):
        df = _make_firms_df().drop(columns=["latitude", "confidence"])
        warnings = validate_firms(df)
        assert any("latitude" in w for w in warnings)
        assert any("confidence" in w for w in warnings)

    def test_latitude_out_of_range(self):
        df = _make_firms_df(latitude=[100.0])
        warnings = validate_firms(df)
        assert any("latitude" in w for w in warnings)

    def test_latitude_negative_out_of_range(self):
        df = _make_firms_df(latitude=[-100.0])
        warnings = validate_firms(df)
        assert any("latitude" in w for w in warnings)

    def test_longitude_out_of_range(self):
        df = _make_firms_df(longitude=[200.0])
        warnings = validate_firms(df)
        assert any("longitude" in w for w in warnings)

    def test_longitude_negative_out_of_range(self):
        df = _make_firms_df(longitude=[-200.0])
        warnings = validate_firms(df)
        assert any("longitude" in w for w in warnings)

    def test_negative_frp_warns(self):
        df = _make_firms_df(frp=[-5.0])
        warnings = validate_firms(df)
        assert any("frp" in w.lower() for w in warnings)

    def test_unparseable_acq_date_warns(self):
        df = _make_firms_df(acq_date=["not-a-date"])
        warnings = validate_firms(df)
        assert any("acq_date" in w for w in warnings)

    def test_valid边界值_no_warnings(self):
        df = _make_firms_df(n=2, latitude=[-90, 90], longitude=[-180, 180], frp=[0.0, 100.0])
        assert validate_firms(df) == []

    def test_returns_list(self):
        df = _make_firms_df()
        assert isinstance(validate_firms(df), list)

    def test_multiple_warnings(self):
        df = _make_firms_df(latitude=[100.0], frp=[-1.0])
        warnings = validate_firms(df)
        assert len(warnings) >= 2


# ---------------------------------------------------------------------------
# validate_enriched_clc
# ---------------------------------------------------------------------------

class TestValidateEnrichedClc:
    def test_no_clc_column_no_warnings(self):
        df = _make_firms_df()
        assert validate_enriched_clc(df) == []

    def test_clc_class_all_present(self):
        df = _make_firms_df(n=3, clc_class=[1, 2, 3])
        assert validate_enriched_clc(df) == []

    def test_clc_class_has_nan(self):
        df = _make_firms_df(n=3, clc_class=[1, np.nan, 3])
        warnings = validate_enriched_clc(df)
        assert len(warnings) == 1
        assert "clc_class" in warnings[0]
        assert "1" in warnings[0]  # 1 missing row

    def test_clc_class_all_nan(self):
        df = _make_firms_df(n=2, clc_class=[np.nan, np.nan])
        warnings = validate_enriched_clc(df)
        assert len(warnings) == 1
        assert "2" in warnings[0]

    def test_empty_df(self):
        df = pd.DataFrame(columns=["clc_class"])
        assert validate_enriched_clc(df) == []


# ---------------------------------------------------------------------------
# validate_enriched_openmeteo
# ---------------------------------------------------------------------------

class TestValidateEnrichedOpenmeteo:
    def test_no_weather_columns_no_warnings(self):
        df = _make_firms_df()
        assert validate_enriched_openmeteo(df) == []

    def test_weather_all_present(self):
        df = _make_firms_df(
            temperature_2m=[25.0],
            relative_humidity_2m=[60.0],
            wind_speed_10m=[5.0],
        )
        assert validate_enriched_openmeteo(df) == []

    def test_temperature_missing(self):
        df = _make_firms_df(temperature_2m=[np.nan])
        warnings = validate_enriched_openmeteo(df)
        assert len(warnings) == 1
        assert "temperature_2m" in warnings[0]

    def test_multiple_weather_missing(self):
        df = _make_firms_df(
            temperature_2m=[np.nan],
            relative_humidity_2m=[np.nan],
            wind_speed_10m=[5.0],
        )
        warnings = validate_enriched_openmeteo(df)
        assert len(warnings) == 2

    def test_partial_weather_columns(self):
        df = _make_firms_df(temperature_2m=[25.0])
        assert validate_enriched_openmeteo(df) == []

    def test_empty_df(self):
        df = pd.DataFrame(columns=["temperature_2m", "relative_humidity_2m", "wind_speed_10m"])
        assert validate_enriched_openmeteo(df) == []


# ---------------------------------------------------------------------------
# validate_enriched (wrapper)
# ---------------------------------------------------------------------------

class TestValidateEnriched:
    def test_calls_all_three(self):
        df = _make_firms_df()
        assert validate_enriched(df) == []

    def test_firms_warning_included(self):
        df = _make_firms_df(latitude=[100.0])
        warnings = validate_enriched(df)
        assert any("latitude" in w for w in warnings)

    def test_clc_warning_included(self):
        df = _make_firms_df(clc_class=[np.nan])
        warnings = validate_enriched(df)
        assert any("clc_class" in w for w in warnings)


# ---------------------------------------------------------------------------
# Integration — real enriched dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def enriched_df() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "enriched" / "firms_spain_enriched.csv"
    return pd.read_csv(path)


class TestAgainstRealData:
    def test_validate_firms_no_warnings(self, enriched_df):
        warnings = validate_firms(enriched_df)
        assert warnings == []

    def test_validate_enriched_clc_detects_gaps(self, enriched_df):
        """Real dataset has rows with missing clc_class — verify detection."""
        warnings = validate_enriched_clc(enriched_df)
        assert len(warnings) == 1
        assert "clc_class" in warnings[0]
        assert "enrichment gap" in warnings[0]

    def test_validate_enriched_openmeteo_skipped(self, enriched_df):
        """Weather columns not present yet — no warnings expected."""
        warnings = validate_enriched_openmeteo(enriched_df)
        assert warnings == []
