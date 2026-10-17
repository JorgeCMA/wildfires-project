"""Contrato del cargador de días de fondo (Plan A, modelo supervisado).

El cargador prefiere el CSV de fondo terminado; si no existe, cae en el
parcial y devuelve solo las filas con ``temperature_2m`` relleno.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wildfire.enrichment.weather_batch import (
    BACKGROUND_COMPLETE_CSV,
    BACKGROUND_PARTIAL_CSV,
    load_background_weather,
)


def _background_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "latitude": [40.0, 41.0, 42.0],
            "longitude": [-3.0, -4.0, -5.0],
            "acq_date": ["2023-06-01", "2023-06-02", "2023-06-03"],
            "temperature_2m": [25.0, None, 30.0],
        }
    )


class TestLoadBackgroundWeather:
    def test_complete_csv_preferred(self, tmp_path):
        complete = tmp_path / "background_weather.csv"
        partial = tmp_path / "background_weather_partial.csv"
        complete.write_text(_background_df().to_csv(index=False), encoding="utf-8")
        partial.write_text(_background_df().to_csv(index=False), encoding="utf-8")

        df = load_background_weather(complete, partial)

        # El terminado se devuelve tal cual: 3 filas, incluida la de NaN.
        assert len(df) == 3

    def test_partial_fallback_keeps_only_filled_rows(self, tmp_path):
        complete = tmp_path / "background_weather.csv"
        partial = tmp_path / "background_weather_partial.csv"
        partial.write_text(_background_df().to_csv(index=False), encoding="utf-8")

        df = load_background_weather(complete, partial)

        assert len(df) == 2
        assert df["temperature_2m"].notna().all()
        # Índice posicional limpio (sin huecos del dropna).
        assert list(df.index) == [0, 1]

    def test_missing_both_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="neither"):
            load_background_weather(tmp_path / "a.csv", tmp_path / "b.csv")

    def test_partial_without_weather_column_raises(self, tmp_path):
        partial = tmp_path / "partial.csv"
        partial.write_text("latitude,longitude\n40,-3\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="temperature_2m"):
            load_background_weather(tmp_path / "complete.csv", partial)

    def test_partial_all_unfilled_raises(self, tmp_path):
        partial = tmp_path / "partial.csv"
        df = _background_df().assign(temperature_2m=None)
        partial.write_text(df.to_csv(index=False), encoding="utf-8")

        with pytest.raises(RuntimeError, match="has not produced data"):
            load_background_weather(tmp_path / "complete.csv", partial)


class TestConfiguredPaths:
    def test_defaults_point_at_synthetic_background_csvs(self):
        assert (
            BACKGROUND_PARTIAL_CSV.name == "firms_spain_background_weather_partial.csv"
        )
        assert BACKGROUND_COMPLETE_CSV.name == "firms_spain_background_weather.csv"
        assert BACKGROUND_PARTIAL_CSV.parent == BACKGROUND_COMPLETE_CSV.parent

    def test_real_partial_loads_filled_rows(self):
        if not BACKGROUND_PARTIAL_CSV.exists():
            pytest.skip("background partial CSV not on disk")
        df = load_background_weather()
        assert "temperature_2m" in df.columns
        assert df["temperature_2m"].notna().all()
        assert not df.empty
