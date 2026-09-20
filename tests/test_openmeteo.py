"""Tests for Open-Meteo historical weather API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from wildfire.data.openmeteo import (
    OPEN_METEO_BASE,
    _default_variables,
    fetch_weather,
)


# ---------------------------------------------------------------------------
# _default_variables
# ---------------------------------------------------------------------------

class TestDefaultVariables:
    def test_returns_list(self):
        result = _default_variables()
        assert isinstance(result, list)

    def test_all_strings(self):
        for var in _default_variables():
            assert isinstance(var, str)

    def test_contains_core_variables(self):
        defaults = _default_variables()
        assert "temperature_2m" in defaults
        assert "relative_humidity_2m" in defaults
        assert "wind_speed_10m" in defaults

    def test_at_least_four_variables(self):
        assert len(_default_variables()) >= 4


# ---------------------------------------------------------------------------
# fetch_weather — unit tests with mocked requests
# ---------------------------------------------------------------------------

class TestFetchWeather:
    def test_calls_correct_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"hourly": {"time": [], "temperature_2m": []}}
        mock_response.raise_for_status = MagicMock()

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response) as mock_get:
            fetch_weather(
                latitude=40.0,
                longitude=-3.0,
                start_date="2023-07-01",
                end_date="2023-07-01",
            )
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            assert args[0] == OPEN_METEO_BASE

    def test_passes_parameters(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"hourly": {}}
        mock_response.raise_for_status = MagicMock()

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response) as mock_get:
            fetch_weather(
                latitude=40.0,
                longitude=-3.0,
                start_date="2023-07-01",
                end_date="2023-07-01",
                hourly_variables=["temperature_2m", "wind_speed_10m"],
            )
            _, kwargs = mock_get.call_args
            params = kwargs["params"]
            assert params["latitude"] == 40.0
            assert params["longitude"] == -3.0
            assert params["start_date"] == "2023-07-01"
            assert params["end_date"] == "2023-07-01"
            assert "temperature_2m" in params["hourly"]
            assert "wind_speed_10m" in params["hourly"]

    def test_default_variables_used(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"hourly": {}}
        mock_response.raise_for_status = MagicMock()

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response) as mock_get:
            fetch_weather(
                latitude=40.0,
                longitude=-3.0,
                start_date="2023-07-01",
                end_date="2023-07-01",
            )
            _, kwargs = mock_get.call_args
            params = kwargs["params"]
            default_vars = _default_variables()
            for var in default_vars:
                assert var in params["hourly"]

    def test_returns_dict(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"hourly": {"time": ["2023-07-01T00:00"], "temperature_2m": [20.0]}}
        mock_response.raise_for_status = MagicMock()

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response):
            result = fetch_weather(
                latitude=40.0,
                longitude=-3.0,
                start_date="2023-07-01",
                end_date="2023-07-01",
            )
            assert isinstance(result, dict)
            assert "hourly" in result

    def test_http_error_raises(self):
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                fetch_weather(
                    latitude=40.0,
                    longitude=-3.0,
                    start_date="2023-07-01",
                    end_date="2023-07-01",
                )

    def test_timeout_passed(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"hourly": {}}
        mock_response.raise_for_status = MagicMock()

        with patch("wildfire.data.openmeteo.requests.get", return_value=mock_response) as mock_get:
            fetch_weather(
                latitude=40.0,
                longitude=-3.0,
                start_date="2023-07-01",
                end_date="2023-07-01",
            )
            _, kwargs = mock_get.call_args
            assert kwargs["timeout"] == 30


# ---------------------------------------------------------------------------
# Integration — real API call (slow, marked)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFetchWeatherIntegration:
    def test_real_api_call(self):
        result = fetch_weather(
            latitude=40.4,
            longitude=-3.7,
            start_date="2023-07-01",
            end_date="2023-07-01",
            hourly_variables=["temperature_2m"],
        )
        assert isinstance(result, dict)
        assert "hourly" in result
        hourly = result["hourly"]
        assert "time" in hourly
        assert "temperature_2m" in hourly
        assert len(hourly["time"]) > 0
