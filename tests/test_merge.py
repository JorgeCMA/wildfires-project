"""Tests para la fusión VIIRS+MODIS (`wildfire.enrichment.merge_sensors`)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from wildfire.enrichment.merge_sensors import (
    load_merged,
    merge_viirs_modis,
    save_merged,
)


def _make_mixed_df() -> pd.DataFrame:
    """2 filas VIIRS (categórica) + 2 MODIS (numérica)."""
    return pd.DataFrame(
        {
            "latitude": [40.0, 41.0, 42.0, 43.0],
            "longitude": [-3.0, -3.0, -3.0, -3.0],
            "acq_date": ["2023-07-01"] * 4,
            "acq_time": [1200] * 4,
            "sensor": ["viirs_snpp", "viirs_snpp", "modis", "modis"],
            "confidence": ["h", "l", "90", "20"],
            "brightness": [300.0] * 4,
            "frp": [10.0] * 4,
        }
    )


def _tmp_config(tmp_path: Path):
    """Redirige output.merged a tmp (no toca data/processed real)."""
    return {"output": {"merged": str(tmp_path)}}


class TestMergeViirsModis:
    @patch("wildfire.enrichment.merge_sensors.load_all_firms")
    def test_adds_unified_confidence(self, mock_load):
        mock_load.return_value = _make_mixed_df()

        out = merge_viirs_modis()

        assert out["confidence_cat"].tolist() == ["h", "l", "h", "l"]
        assert out["confidence_num"].tolist() == [85.0, 15.0, 90.0, 20.0]
        assert out["confidence_og_num"].notna().tolist() == [False, False, True, True]
        assert out["confidence_og_cat"].notna().tolist() == [True, True, False, False]

    @patch("wildfire.enrichment.merge_sensors.load_all_firms")
    def test_empty_input_returns_empty(self, mock_load):
        mock_load.return_value = pd.DataFrame()
        assert merge_viirs_modis().empty

    @patch("wildfire.enrichment.merge_sensors.load_all_firms")
    def test_validation_warnings_are_printed(self, mock_load, capsys):
        df = _make_mixed_df()
        df.loc[0, "latitude"] = 100.0
        mock_load.return_value = df

        merge_viirs_modis()

        assert "latitude" in capsys.readouterr().out

    @patch("wildfire.enrichment.merge_sensors.load_all_firms")
    def test_clean_input_prints_nothing(self, mock_load, capsys):
        mock_load.return_value = _make_mixed_df()

        merge_viirs_modis()

        assert capsys.readouterr().out == ""


class TestSaveLoadMerged:
    def test_roundtrip(self, tmp_path, monkeypatch):
        import wildfire.enrichment.merge_sensors as ms

        monkeypatch.setattr(ms, "load_config", lambda: _tmp_config(tmp_path))
        df = _make_mixed_df()

        path = save_merged(df)
        assert path.exists()

        loaded = load_merged()
        pd.testing.assert_frame_equal(loaded, df)

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        import wildfire.enrichment.merge_sensors as ms

        monkeypatch.setattr(ms, "load_config", lambda: _tmp_config(tmp_path))
        with pytest.raises(FileNotFoundError, match="Merged file not found"):
            load_merged()

    def test_sensor_folders_match_config(self):
        """El dict del código y firms.sensor_folders no deben divergir."""
        from wildfire.config import load_config
        from wildfire.data.firms import SENSOR_FOLDER_NAMES

        assert load_config()["firms"]["sensor_folders"] == SENSOR_FOLDER_NAMES
