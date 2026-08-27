"""Enrich FIRMS data with Open-Meteo historical weather data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.config import load_config
from wildfire.data.openmeteo import fetch_weather


def enrich_with_weather(
    df: pd.DataFrame,
    hourly_variables: list[str] | None = None,
) -> pd.DataFrame:
    """Add weather data from Open-Meteo to each FIRMS row.

    For each fire detection, the code fetches hourly weather at the
    detection coordinates for the acquisition date. The weather values
    at the acquisition hour are used.

    Parameters
    ----------
    df:
        FIRMS DataFrame with ``latitude``, ``longitude``, ``acq_date``,
        and ``acq_time`` columns.
    hourly_variables:
        List of Open-Meteo hourly variable names. Defaults to core fire
        weather variables.

    Returns
    -------
    pd.DataFrame
        DataFrame with added weather columns.
    """
    df = df.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")

    weather_records: list[dict] = []

    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        date = row["acq_date"]

        if pd.isna(lat) or pd.isna(lon) or pd.isna(date):
            weather_records.append({})
            continue

        date_str = date.strftime("%Y-%m-%d")

        try:
            weather = fetch_weather(
                latitude=lat,
                longitude=lon,
                start_date=date_str,
                end_date=date_str,
                hourly_variables=hourly_variables,
            )
        except Exception:
            weather_records.append({})
            continue

        hourly = weather.get("hourly", {})
        time_labels = hourly.get("time", [])

        acq_time = str(row.get("acq_time", "")).zfill(4)
        hour_str = f"{date_str}T{acq_time[:2]}:00"

        record: dict = {}
        for var in hourly.get("time", []):
            pass

        for key, values in hourly.items():
            if key == "time":
                continue
            if hour_str in time_labels:
                idx = time_labels.index(hour_str)
                record[key] = values[idx] if idx < len(values) else None
            elif values:
                record[key] = values[0]
            else:
                record[key] = None

        weather_records.append(record)

    weather_df = pd.DataFrame(weather_records, index=df.index)
    return pd.concat([df, weather_df], axis=1)


def save_weather_enriched(
    df: pd.DataFrame,
    country: str = "Spain",
    year: int | None = None,
) -> Path:
    """Save weather-enriched FIRMS data.

    Parameters
    ----------
    df:
        Weather-enriched FIRMS DataFrame.
    country:
        Country name.
    year:
        If provided, save as ``firms_{country}_{year}_enriched.csv``
        (overwrites CLC enrichment, which should run first).

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    config = load_config()
    out_dir = Path(config["output"]["enriched"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_enriched.csv"
    else:
        filename = f"firms_{country.lower()}_enriched.csv"

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path
