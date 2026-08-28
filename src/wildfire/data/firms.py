"""NASA FIRMS data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.config import load_config

VIIRS_COLUMNS = {
    "latitude": float,
    "longitude": float,
    "bright_ti4": float,
    "scan": float,
    "track": float,
    "acq_date": str,
    "acq_time": str,
    "satellite": str,
    "confidence": str,
    "version": str,
    "bright_ti5": float,
    "frp": float,
    "daynight": str,
}

MODIS_COLUMNS = {
    "latitude": float,
    "longitude": float,
    "brightness": float,
    "scan": float,
    "track": float,
    "acq_date": str,
    "acq_time": str,
    "satellite": str,
    "confidence": str,
    "version": str,
    "bright_t31": float,
    "frp": float,
    "daynight": str,
}

VIIRS_COLS_TO_RENAME = {
    "bright_ti4": "brightness",
    "bright_ti5": "brightness_ir",
}

MODIS_COLS_TO_RENAME = {
    "bright_t31": "brightness_ir",
}

SENSOR_FOLDER_NAMES = {
    "modis": "MODIS",
    "viirs_noaa20": "VIIRS/VIIRS NOAA-20",
    "viirs_snpp": "VIIRS/VIIRS S-NPP",
}


def _firms_dir(country: str, year: int, sensor: str) -> Path:
    """Build the directory path for a FIRMS sensor/year/country."""
    from wildfire.config import PROJECT_ROOT
    config = load_config()
    sensor_path = SENSOR_FOLDER_NAMES.get(sensor.lower(), sensor)
    return PROJECT_ROOT / config["data"]["raw"] / "firms" / country / str(year) / sensor_path


def _find_csv_in_dir(directory: Path) -> Path | None:
    """Find the first CSV file in a directory."""
    if not directory.exists():
        return None
    csvs = list(directory.glob("*.csv"))
    return csvs[0] if csvs else None


def load_firms(
    country: str = "Spain",
    year: int = 2023,
    sensor: str = "viirs_snpp",
) -> pd.DataFrame:
    """Load a single FIRMS CSV file and normalise column names.

    Parameters
    ----------
    country:
        Country name (must match folder name under ``data/raw/firms/``).
    year:
        Four-digit year.
    sensor:
        One of ``"modis"``, ``"viirs_snpp"``, ``"viirs_noaa20"``.

    Returns
    -------
    pd.DataFrame
        Normalised DataFrame with a ``sensor`` column added.

    Raises
    ------
    ValueError
        If ``sensor`` is not recognised.
    FileNotFoundError
        If the CSV file does not exist.
    """
    sensor = sensor.lower()

    if sensor not in SENSOR_FOLDER_NAMES:
        valid = ", ".join(sorted(SENSOR_FOLDER_NAMES.keys()))
        raise ValueError(f"Unknown sensor: {sensor!r}. Use one of: {valid}")

    directory = _firms_dir(country, year, sensor)
    csv_path = _find_csv_in_dir(directory)

    if csv_path is None:
        raise FileNotFoundError(f"No FIRMS CSV found in {directory}")

    df = pd.read_csv(csv_path)

    df["sensor"] = sensor
    df["country"] = country
    df["year"] = year

    if sensor.startswith("viirs"):
        df.rename(columns=VIIRS_COLS_TO_RENAME, inplace=True)
    elif sensor == "modis":
        df.rename(columns=MODIS_COLS_TO_RENAME, inplace=True)

    return df


def load_all_firms(
    country: str = "Spain",
    years: list[int] | None = None,
    sensors: list[str] | None = None,
) -> pd.DataFrame:
    """Load FIRMS data for all specified countries, years, and sensors.

    Parameters
    ----------
    country:
        Country name.
    years:
        List of years. Defaults to configured years.
    sensors:
        List of sensors. Defaults to ``["modis", "viirs_snpp", "viirs_noaa20"]``.

    Returns
    -------
    pd.DataFrame
        Concatenated FIRMS data from all specified combinations.
    """
    if years is None:
        config = load_config()
        years = config["firms"]["years"]
    if sensors is None:
        sensors = ["modis", "viirs_snpp", "viirs_noaa20"]

    frames: list[pd.DataFrame] = []
    for year in years:
        for sensor in sensors:
            try:
                df = load_firms(country=country, year=year, sensor=sensor)
                frames.append(df)
            except FileNotFoundError:
                continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def list_available_firms(country: str = "Spain") -> list[dict[str, str | int]]:
    """List which FIRMS files exist on disk.

    Walks the folder hierarchy under ``data/raw/firms/{country}/{year}/`` and
    derives the sensor key from the folder name rather than parsing filenames.

    Returns
    -------
    list[dict]
        Each dict has keys ``country``, ``year``, ``sensor``, ``path``.
    """
    config = load_config()
    base_dir = Path(config["data"]["raw"]) / "firms" / country
    results: list[dict[str, str | int]] = []

    if not base_dir.exists():
        return results

    folder_to_sensor = {v.lower(): k for k, v in SENSOR_FOLDER_NAMES.items()}

    for year_dir in sorted(base_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        try:
            year = int(year_dir.name)
        except ValueError:
            continue
        for csv_path in sorted(year_dir.rglob("*.csv")):
            # csv_path is like .../{year}/MODIS/file.csv or .../{year}/VIIRS/VIIRS S-NPP/file.csv
            # Use relative path from year_dir to match SENSOR_FOLDER_NAMES keys
            rel_dir = csv_path.parent.relative_to(year_dir).as_posix().lower()
            sensor_key = folder_to_sensor.get(rel_dir, rel_dir)
            results.append({
                "country": country,
                "year": year,
                "sensor": sensor_key,
                "path": str(csv_path),
            })

    return results
