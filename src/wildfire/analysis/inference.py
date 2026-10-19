"""Live point scoring: enrich and engineer the predictors for one coordinate.

The batch pipeline (notebook 01) turns raw detections into the model's daily
predictors by chaining weather, CLC, CCAA budgets and notebook-01b engineering.
This module replays that exact chain for a *single* point-at-date request so the
demo can score an arbitrary location (Phase 2 of the Plan-D demo).

It deliberately reuses the production building blocks (``fetch_weather``,
``interpolate_hourly_fields``, ``enrich_with_clc``, ``assign_ccaa``,
``add_ccaa_budget_sums``, ``add_engineered_features``) instead of re-deriving
any formula, so a live point lands in the same feature space as the training
frame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from wildfire.data.ccaa import add_ccaa_budget_sums
from wildfire.data.openmeteo import fetch_weather
from wildfire.enrichment.clc_enrichment import enrich_with_clc
from wildfire.enrichment.weather_batch import (
    interpolate_hourly_fields,
    weather_fields,
)
from wildfire.geo.regions import assign_ccaa
from wildfire.processing.feature_engineering import add_engineered_features
from wildfire.processing.occurrence import occurrence_features

#: Signature of the Open-Meteo fetcher (kept loose for monkeypatching in tests).
WeatherFetcher = Callable[..., dict[str, Any]]


def json_to_hourly_records(payload: dict[str, Any]) -> list[dict]:
    """Turn a plain Open-Meteo JSON response into the record list the pipeline uses.

    The archive API returns ``{"hourly": {"time": [...], "<var>": [...]}}``;
    :func:`wildfire.enrichment.weather_batch.interpolate_hourly_fields` expects
    one dict per hour with a ``date`` key plus a key per variable.
    """
    if "hourly" not in payload:
        raise KeyError("Open-Meteo response has no 'hourly' block")
    hourly = payload["hourly"]
    times = list(hourly["time"])
    variables = [key for key in hourly if key != "time"]
    return [
        {"date": times[i], **{var: hourly[var][i] for var in variables}}
        for i in range(len(times))
    ]


def build_point_features(
    latitude: float,
    longitude: float,
    date: str | pd.Timestamp,
    hour: int = 12,
    minute: int = 0,
    *,
    weather_fetcher: WeatherFetcher = fetch_weather,
) -> pd.DataFrame:
    """Enriched + engineered predictor frame (one row) for a live point request.

    Parameters
    ----------
    latitude, longitude:
        Point coordinates in WGS84 degrees.
    date:
        Acquisition date; anything ``pandas.Timestamp`` can parse (``YYYY-MM-DD``).
    hour, minute:
        Time of day (UTC), defaulting to midday. Internally packed as the FIRMS
        ``HHMM`` ``acq_time`` so the hourly weather is interpolated the same way
        as batch rows.
    weather_fetcher:
        Callable matching :func:`wildfire.data.openmeteo.fetch_weather`; injected
        in tests to avoid network calls.

    Returns
    -------
    pd.DataFrame
        Single-row frame with exactly ``EVERYDAY_FEATURES`` columns (the model's
        predictor contract) plus the index of a single row.

    Raises
    ------
    ValueError
        If ``hour``/``minute`` are outside their ranges.
    KeyError
        If the weather response is malformed or a predictor cannot be built
        (fails closed rather than scoring on partial features).
    """
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be in 0..23, got {hour!r}")
    if not 0 <= minute <= 59:
        raise ValueError(f"minute must be in 0..59, got {minute!r}")

    acq_date = pd.Timestamp(date).strftime("%Y-%m-%d")
    acq_time = int(f"{hour:02d}{minute:02d}")

    payload = weather_fetcher(
        latitude=float(latitude),
        longitude=float(longitude),
        start_date=acq_date,
        end_date=acq_date,
        hourly_variables=weather_fields(),
    )
    records = json_to_hourly_records(payload)
    weather = interpolate_hourly_fields(records, acq_time)

    frame = pd.DataFrame(
        {
            "latitude": [float(latitude)],
            "longitude": [float(longitude)],
            "acq_date": [acq_date],
            "acq_time": [acq_time],
            **{field: [value] for field, value in weather.items()},
        }
    )

    frame = enrich_with_clc(frame)
    frame = assign_ccaa(frame)
    frame = add_ccaa_budget_sums(frame)
    frame = add_engineered_features(frame)
    return occurrence_features(frame)


__all__ = [
    "WeatherFetcher",
    "build_point_features",
    "json_to_hourly_records",
]
