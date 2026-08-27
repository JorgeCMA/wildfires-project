"""Basic tests for FIRMS data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wildfire.data.firms import (
    SENSOR_FOLDER_NAMES,
    _firms_dir,
    _find_csv_in_dir,
    load_firms,
    load_all_firms,
    list_available_firms,
)
from wildfire.config import PROJECT_ROOT


class TestSensorFolderNames:
    def test_modis_mapping(self):
        assert SENSOR_FOLDER_NAMES["modis"] == "MODIS"

    def test_viirs_snpp_mapping(self):
        assert SENSOR_FOLDER_NAMES["viirs_snpp"] == "VIIRS/VIIRS S-NPP"

    def test_viirs_noaa20_mapping(self):
        assert SENSOR_FOLDER_NAMES["viirs_noaa20"] == "VIIRS/VIIRS NOAA-20"

    def test_all_keys_are_lowercase(self):
        for key in SENSOR_FOLDER_NAMES:
            assert key == key.lower(), f"Key {key!r} is not lowercase"


class TestFirmsDir:
    def test_modis_path(self):
        path = _firms_dir("Spain", 2023, "modis")
        assert path.name == "MODIS"
        assert "2023" in str(path)

    def test_viirs_snpp_path(self):
        path = _firms_dir("Spain", 2023, "viirs_snpp")
        assert path.name == "VIIRS S-NPP"
        assert path.parent.name == "VIIRS"

    def test_viirs_noaa20_path(self):
        path = _firms_dir("Spain", 2023, "viirs_noaa20")
        assert path.name == "VIIRS NOAA-20"
        assert path.parent.name == "VIIRS"


class TestFindCsvInDir:
    def test_finds_csv(self, tmp_path: Path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        result = _find_csv_in_dir(tmp_path)
        assert result == csv_file

    def test_returns_none_when_empty(self, tmp_path: Path):
        result = _find_csv_in_dir(tmp_path)
        assert result is None

    def test_returns_none_when_no_csv(self, tmp_path: Path):
        (tmp_path / "data.txt").write_text("hello")
        result = _find_csv_in_dir(tmp_path)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self):
        result = _find_csv_in_dir(Path("/nonexistent"))
        assert result is None

    def test_returns_first_csv(self, tmp_path: Path):
        (tmp_path / "a.csv").write_text("a")
        (tmp_path / "b.csv").write_text("b")
        result = _find_csv_in_dir(tmp_path)
        assert result is not None
        assert result.suffix == ".csv"


class TestLoadFirms:
    def test_load_viirs_snpp(self):
        df = load_firms(country="Spain", year=2023, sensor="viirs_snpp")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "sensor" in df.columns
        assert (df["sensor"] == "viirs_snpp").all()
        assert "latitude" in df.columns
        assert "longitude" in df.columns
        assert "acq_date" in df.columns
        assert "brightness" in df.columns

    def test_load_viirs_noaa20(self):
        df = load_firms(country="Spain", year=2023, sensor="viirs_noaa20")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert (df["sensor"] == "viirs_noaa20").all()

    def test_load_modis(self):
        df = load_firms(country="Spain", year=2023, sensor="modis")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert (df["sensor"] == "modis").all()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_firms(country="Spain", year=9999, sensor="modis")

    def test_load_invalid_sensor_raises(self):
        with pytest.raises(ValueError):
            load_firms(country="Spain", year=2023, sensor="invalid_sensor")


class TestLoadAllFirms:
    def test_load_all_returns_dataframe(self):
        df = load_all_firms(country="Spain", years=[2023])
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_all_has_multiple_sensors(self):
        df = load_all_firms(country="Spain", years=[2023])
        sensors = df["sensor"].unique()
        assert len(sensors) >= 2


class TestListAvailableFirms:
    def test_list_returns_list(self):
        result = list_available_firms(country="Spain")
        assert isinstance(result, list)

    def test_list_has_entries(self):
        result = list_available_firms(country="Spain")
        assert len(result) > 0

    def test_list_entry_structure(self):
        result = list_available_firms(country="Spain")
        entry = result[0]
        assert "country" in entry
        assert "year" in entry
        assert "sensor" in entry
        assert "path" in entry

    def test_list_sensors_are_correct_keys(self):
        result = list_available_firms(country="Spain")
        valid_keys = set(SENSOR_FOLDER_NAMES.keys())
        for entry in result:
            assert entry["sensor"] in valid_keys, (
                f"Sensor {entry['sensor']!r} not in SENSOR_FOLDER_NAMES"
            )
