"""Open-Meteo Historical Weather API client."""

from __future__ import annotations

from typing import Any

import requests

from wildfire.config import load_config

OPEN_METEO_BASE = "https://archive-api.open-meteo.com/v1/archive"


def _default_variables() -> list[str]:
    """Return the default hourly weather variables from config."""
    config = load_config()
    return config.get("openmeteo", {}).get("hourly_variables", [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
        "shortwave_radiation",
    ])


def fetch_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_variables: list[str] | None = None,
) -> dict[str, Any]:
    """Fetch historical weather data from Open-Meteo.

    Parameters
    ----------
    latitude:
        Latitude in decimal degrees (WGS84).
    longitude:
        Longitude in decimal degrees (WGS84).
    start_date:
        Start date as ``"YYYY-MM-DD"``.
    end_date:
        End date as ``"YYYY-MM-DD"``.
    hourly_variables:
        List of Open-Meteo hourly variable names. Defaults to core fire
        weather variables defined in ``configs/project.yaml``.

    Returns
    -------
    dict
        The JSON response from Open-Meteo, containing at least an
        ``hourly`` key with ``time`` and requested variable arrays.

    Raises
    ------
    requests.HTTPError
        If the API request fails.
    """
    if hourly_variables is None:
        hourly_variables = _default_variables()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
    }

    response = requests.get(OPEN_METEO_BASE, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
