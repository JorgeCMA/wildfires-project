"""Unit tests for the engineered features shared by batch and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.feature_engineering import (
    CCAA_HOLIDAY_CODES,
    SUP_FORESTAL_HA,
    add_anthropogenic_features,
    add_budget_features,
    add_calendar_features,
    add_cyclic_features,
    add_engineered_features,
    add_meteo_indices,
)


class TestCalendarFeatures:
    def test_month_dayofyear_weekend(self):
        df = pd.DataFrame(
            {"acq_date": ["2023-01-06", "2023-01-07", "2023-01-10"]}  # Fri, Sat, Tue
        )
        out = add_calendar_features(df)

        assert out["month"].tolist() == [1, 1, 1]
        assert out["day_of_year"].tolist() == [6, 7, 10]
        assert out["es_fin_semana"].tolist() == [0, 1, 0]
        assert pd.api.types.is_datetime64_any_dtype(out["acq_date"])

    def test_custom_date_col(self):
        df = pd.DataFrame({"day": ["2023-07-01"]})  # Saturday
        out = add_calendar_features(df, date_col="day")

        assert out["es_fin_semana"].iloc[0] == 1


class TestAnthropogenicFeatures:
    def test_epiphany_is_holiday_in_madrid(self):
        df = pd.DataFrame(
            {"acq_date": ["2023-01-06", "2023-01-10"], "ccaa": ["Madrid", "Madrid"]}
        )
        out = add_anthropogenic_features(df)

        # 6 Jan (Epiphany) is a holiday -> high anthropogenic risk; 10 Jan is not.
        assert out["es_festivo"].tolist() == [1, 0]
        assert out["riesgo_antropico_alto"].tolist() == [1, 0]

    def test_weekend_counts_even_without_holiday(self):
        df = pd.DataFrame(
            {"acq_date": ["2023-07-01"], "ccaa": ["Madrid"]}  # Saturday
        )
        out = add_anthropogenic_features(df)

        assert out["es_festivo"].iloc[0] == 0
        assert out["riesgo_antropico_alto"].iloc[0] == 1

    def test_unknown_region_has_no_holiday(self):
        out = add_anthropogenic_features(
            pd.DataFrame({"acq_date": ["2023-01-06"], "ccaa": ["Wonderland"]})
        )

        assert out["es_festivo"].iloc[0] == 0

    def test_holiday_codes_cover_every_budget_region(self):
        assert set(CCAA_HOLIDAY_CODES) == set(SUP_FORESTAL_HA)


class TestCyclicFeatures:
    def test_month_sin_cos(self):
        out = add_cyclic_features(pd.DataFrame({"month": [1], "day_of_year": [1]}))

        assert out["month_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi / 12))
        assert out["month_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi / 12))
        assert out["day_of_year_sin"].iloc[0] == pytest.approx(
            np.sin(2 * np.pi / 365.25)
        )


class TestBudgetFeatures:
    def test_ratios_and_hectares(self):
        df = pd.DataFrame(
            {
                "ccaa": ["Madrid", "Madrid"],
                "prevention_prev3": [100.0, 200.0],
                "extinction_prev3": [50.0, 50.0],
            }
        )
        out = add_budget_features(df, lookback_years=3)

        assert out["sup_forestal_ha"].tolist() == [SUP_FORESTAL_HA["Madrid"]] * 2
        assert out["ratio_prevencion_extincion"].iloc[0] == pytest.approx(
            100 / 50.00001
        )
        assert out["euros_prevencion_por_ha"].iloc[0] == pytest.approx(
            100 * 1_000_000 / SUP_FORESTAL_HA["Madrid"]
        )

    def test_missing_columns_raise(self):
        with pytest.raises(KeyError):
            add_budget_features(pd.DataFrame({"ccaa": ["Madrid"]}), lookback_years=3)


class TestMeteoIndices:
    def test_formulas(self):
        df = pd.DataFrame(
            {
                "vapour_pressure_deficit": [3.0],
                "wind_speed_10m": [10.0],
                "soil_moisture_0_to_7cm": [0.2],
                "temperature_2m": [30.0],
                "relative_humidity_2m": [20.0],
            }
        )
        out = add_meteo_indices(df)

        propagation = 3.0 * 10.0
        risk = propagation / (0.2 + 0.01)
        assert out["potencial_propagacion"].iloc[0] == pytest.approx(propagation)
        assert out["indice_riesgo_meteo"].iloc[0] == pytest.approx(risk)
        assert out["estres_termico_vegetacion"].iloc[0] == pytest.approx(30 / 21)
        assert out["indice_riesgo_meteo_log"].iloc[0] == pytest.approx(np.log1p(risk))

    def test_missing_weather_raises(self):
        with pytest.raises(KeyError):
            add_meteo_indices(pd.DataFrame({"temperature_2m": [20.0]}))


class TestEngineeredEndToEnd:
    def test_full_chain_adds_all_columns(self):
        df = pd.DataFrame(
            {
                "acq_date": ["2023-07-15"],
                "ccaa": ["Madrid"],
                "prevention_prev3": [100.0],
                "extinction_prev3": [50.0],
                "vapour_pressure_deficit": [3.0],
                "wind_speed_10m": [10.0],
                "soil_moisture_0_to_7cm": [0.2],
                "temperature_2m": [30.0],
                "relative_humidity_2m": [20.0],
            }
        )
        out = add_engineered_features(df, lookback_years=3)

        for column in (
            "month",
            "day_of_year",
            "es_fin_semana",
            "es_festivo",
            "riesgo_antropico_alto",
            "month_sin",
            "month_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "sup_forestal_ha",
            "ratio_prevencion_extincion",
            "euros_prevencion_por_ha",
            "euros_extincion_por_ha",
            "potencial_propagacion",
            "indice_riesgo_meteo",
            "indice_riesgo_meteo_log",
            "estres_termico_vegetacion",
        ):
            assert column in out.columns, column
        assert out["sup_forestal_ha"].iloc[0] == SUP_FORESTAL_HA["Madrid"]
