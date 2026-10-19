"""Tests for the live point scorer's feature assembly (``analysis/inference.py``).

The weather fetch is always injected, so nothing here spends Open-Meteo quota.
The contract test reads the local CLC tiles and CCAA boundaries (skipped if
those untracked inputs are absent, like the other real-data tests).
"""

from __future__ import annotations

import pytest

from wildfire.analysis.inference import build_point_features, json_to_hourly_records
from wildfire.config import PROJECT_ROOT
from wildfire.enrichment.weather_batch import weather_fields
from wildfire.processing.occurrence import EVERYDAY_FEATURES

MADRID = (40.4168, -3.7038)
DATE = "2024-07-15"
GEOJSON = PROJECT_ROOT / "data" / "raw" / "ccaa" / "spain-communities.geojson"
CLC_ROOT = PROJECT_ROOT / "data" / "raw" / "clcplus" / "Spain" / "2023-2025"


def _fake_fetcher(hour_values: dict[str, list[float]] | None = None):
    """Fetcher returning 24 hourly rows; each variable grows with the hour."""

    def fetch(latitude, longitude, start_date, end_date, hourly_variables=None):
        fields = list(weather_fields())
        times = [f"{DATE}T{hour:02d}:00" for hour in range(24)]
        hourly = {"time": times}
        for position, field in enumerate(fields):
            hourly[field] = [float(hour) * (position + 1) for hour in range(24)]
        if hour_values:
            hourly.update(hour_values)
        return {"hourly": hourly}

    return fetch


class TestJsonToHourlyRecords:
    def test_transposes_columns_into_records(self):
        payload = {
            "hourly": {
                "time": ["2024-07-15T00:00", "2024-07-15T01:00"],
                "temperature_2m": [1.0, 2.0],
                "wind_speed_10m": [3.0, 4.0],
            }
        }
        records = json_to_hourly_records(payload)

        assert records == [
            {"date": "2024-07-15T00:00", "temperature_2m": 1.0, "wind_speed_10m": 3.0},
            {"date": "2024-07-15T01:00", "temperature_2m": 2.0, "wind_speed_10m": 4.0},
        ]

    def test_missing_hourly_block_raises(self):
        with pytest.raises(KeyError):
            json_to_hourly_records({"latitude": 40.0})


class TestInputValidation:
    def test_bad_hour_raises(self):
        with pytest.raises(ValueError):
            build_point_features(
                *MADRID, DATE, hour=24, weather_fetcher=_fake_fetcher()
            )

    def test_bad_minute_raises(self):
        with pytest.raises(ValueError):
            build_point_features(
                *MADRID, DATE, minute=60, weather_fetcher=_fake_fetcher()
            )


@pytest.mark.skipif(
    not (GEOJSON.exists() and CLC_ROOT.exists()),
    reason="needs the untracked CCAA geojson and CLC tiles",
)
class TestBuildPointFeatures:
    def test_contract_and_interpolation(self):
        # temperature_2m is the first requested field => value == hour.
        features = build_point_features(
            *MADRID, DATE, hour=12, minute=45, weather_fetcher=_fake_fetcher()
        )

        assert list(features.columns) == EVERYDAY_FEATURES
        assert len(features) == 1
        assert features["ccaa"].iloc[0] == "Madrid"
        assert features["clc_class"].iloc[0] is not None
        # 12:45 -> 25% of 12:00 + 75% of 13:00 = 12.75
        assert features["temperature_2m"].iloc[0] == pytest.approx(12.75)
        assert features["month"].iloc[0] == 7
        assert features["prevention_prev3"].iloc[0] > 0

    def test_noon_default_matches_zero_minutes(self):
        features = build_point_features(
            *MADRID, DATE, hour=12, weather_fetcher=_fake_fetcher()
        )

        assert features["temperature_2m"].iloc[0] == pytest.approx(12.0)
