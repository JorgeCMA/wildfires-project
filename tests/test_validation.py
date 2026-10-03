"""Tests for data quality validation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT, load_config
from wildfire.enrichment.weather_batch import weather_fields
from wildfire.processing.validation import (
    validate_enriched,
    validate_enriched_ccaa,
    validate_enriched_clc,
    validate_enriched_openmeteo,
    validate_firms,
)

WEATHER_FIELDS = weather_fields()
CLC_NEIGHBOR_COLS = [f"clc_class_{s}" for s in ["N", "S", "W", "E", "NW", "NE", "SW", "SE"]]


def _full_clc_df(n=3, **overrides):
    """FIRMS + todas las columnas CLC de la etapa (anillos 1 y 2)."""
    cols = {"clc_class": [1] * n}
    cols.update({c: [1] * n for c in CLC_NEIGHBOR_COLS})
    cols["clc_uniform_surroundings"] = [True] * n
    cols.update(overrides)
    return _make_firms_df(n=n, **cols)


def _full_weather_df(n=1, **overrides):
    """FIRMS + las 15 variables de producción con valor."""
    cols = {f: [1.0] * n for f in WEATHER_FIELDS}
    cols.update(overrides)
    return _make_firms_df(n=n, **cols)


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

    def test_unparseable_acq_date_counts_rows(self):
        df = _make_firms_df(n=3, acq_date=["2023-07-01", "not-a-date", None])
        warnings = validate_firms(df)
        assert any("2" in w and "acq_date" in w for w in warnings)

    def test_string_coords_do_not_crash(self):
        df = _make_firms_df(latitude=["40.0"], longitude=["-3.0"], frp=["10.0"])
        assert validate_firms(df) == []

    def test_missing_latitude_warns(self):
        df = _make_firms_df(latitude=[None])
        warnings = validate_firms(df)
        assert any("latitude" in w for w in warnings)

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
    def test_missing_clc_columns_warn(self):
        # Fail-closed: sin columnas CLC no se puede salir limpio.
        df = _make_firms_df()
        warnings = validate_enriched_clc(df)
        assert any("Missing expected columns" in w for w in warnings)
        assert any("clc_class" in w for w in warnings)

    def test_full_stage_passes_clean(self):
        df = _full_clc_df(n=3)
        assert validate_enriched_clc(df) == []

    def test_clc_class_has_nan(self):
        df = _full_clc_df(n=3, clc_class=[1, np.nan, 3])
        warnings = validate_enriched_clc(df)
        assert len(warnings) == 1
        assert "clc_class" in warnings[0]
        assert "1" in warnings[0]  # 1 missing row

    def test_clc_class_all_nan(self):
        df = _full_clc_df(n=2, clc_class=[np.nan, np.nan])
        warnings = validate_enriched_clc(df)
        assert len(warnings) == 1
        assert "2" in warnings[0]

    def test_neighbor_gap_detected(self):
        df = _full_clc_df(n=2, clc_class_N=[1, np.nan])
        warnings = validate_enriched_clc(df)
        assert len(warnings) == 1
        assert "clc_class_N" in warnings[0]

    def test_empty_df(self):
        df = pd.DataFrame(columns=["clc_class"])
        warnings = validate_enriched_clc(df)
        assert any("Missing expected columns" in w for w in warnings)


# ---------------------------------------------------------------------------
# validate_enriched_openmeteo
# ---------------------------------------------------------------------------

class TestValidateEnrichedOpenmeteo:
    def test_missing_weather_columns_warn(self):
        # Fail-closed: sin columnas de clima no se puede salir limpio.
        df = _make_firms_df()
        warnings = validate_enriched_openmeteo(df)
        assert any("Missing expected columns" in w for w in warnings)
        assert any("temperature_2m" in w for w in warnings)

    def test_all_fifteen_present_passes_clean(self):
        df = _full_weather_df(temperature_2m=[25.0])
        assert validate_enriched_openmeteo(df) == []

    def test_temperature_missing(self):
        df = _full_weather_df(temperature_2m=[np.nan])
        warnings = validate_enriched_openmeteo(df)
        assert len(warnings) == 1
        assert "temperature_2m" in warnings[0]

    def test_multiple_weather_missing(self):
        df = _full_weather_df(
            temperature_2m=[np.nan],
            relative_humidity_2m=[np.nan],
            wind_speed_10m=[5.0],
        )
        warnings = validate_enriched_openmeteo(df)
        assert len(warnings) == 2

    def test_partial_weather_columns_warn_missing(self):
        df = _make_firms_df(temperature_2m=[25.0])
        warnings = validate_enriched_openmeteo(df)
        assert any("Missing expected columns" in w for w in warnings)
        # ...pero la presente y rellena no genera aviso de hueco
        assert not any("missing temperature_2m" in w for w in warnings)

    def test_empty_df(self):
        df = pd.DataFrame(columns=["temperature_2m"])
        warnings = validate_enriched_openmeteo(df)
        assert any("Missing expected columns" in w for w in warnings)


# ---------------------------------------------------------------------------
# validate_enriched (wrapper)
# ---------------------------------------------------------------------------

class TestValidateEnrichedCcaa:
    def test_missing_ccaa_columns_warn(self):
        df = _make_firms_df()
        warnings = validate_enriched_ccaa(df)
        assert any("Missing expected columns" in w for w in warnings)

    def test_full_final_stage_passes_clean(self):
        df = _make_firms_df(
            ccaa=["Madrid"], sum_prevention=[10.0], sum_extinction=[20.0]
        )
        assert validate_enriched_ccaa(df) == []

    def test_nan_sums_are_informational_not_missing(self):
        # Ceuta/Melilla no tienen presupuesto: NaN legítimo, avisa con cuenta.
        df = _make_firms_df(
            n=2,
            ccaa=["Madrid", "Ceuta"],
            sum_prevention=[10.0, np.nan],
            sum_extinction=[20.0, np.nan],
        )
        warnings = validate_enriched_ccaa(df)
        assert len(warnings) == 2
        assert any("sum_prevention" in w and "1" in w for w in warnings)
        assert any("sum_extinction" in w and "1" in w for w in warnings)


class TestValidateEnriched:
    def test_calls_all_four(self):
        df = _make_firms_df()
        warnings = validate_enriched(df)
        assert any("Missing expected columns" in w for w in warnings)

    def test_full_pipeline_stage_passes_clean(self):
        df = _full_weather_df()
        for col, vals in _full_clc_df(n=1).items():
            df[col] = vals
        df["ccaa"] = ["Madrid"]
        df["sum_prevention"] = [10.0]
        df["sum_extinction"] = [20.0]
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
        assert any("clc_class" in w and "enrichment gap" in w for w in warnings)

    def test_validate_enriched_openmeteo_missing_columns_warn(self, enriched_df):
        """Legacy CSV has no weather columns — fail-closed warning expected."""
        warnings = validate_enriched_openmeteo(enriched_df)
        assert any("Missing expected columns" in w for w in warnings)
        assert any("temperature_2m" in w for w in warnings)
