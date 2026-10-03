"""Tests para el módulo de presupuestos CCAA (`wildfire.data.ccaa`)."""

from pathlib import Path

import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT
from wildfire.data.ccaa import (
    EXTINCTION_COL,
    PREVENTION_COL,
    REGION_COL,
    SUM_EXTINCTION_COL,
    SUM_PREVENTION_COL,
    YEAR_COL,
    add_ccaa_budget_sums,
    load_ccaa_budget,
)

# Ruta real del fichero de presupuestos (tracked en git).
REAL_BUDGET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ccaa"
    / "Datos_CCAA_Presupuesto - Datos y Variables Model.csv"
)


def _make_budget(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """Budget sintético: filas de la forma (región, año, prevención, extinción)."""
    return pd.DataFrame(
        {
            REGION_COL: [region for region, *_ in rows],
            YEAR_COL: [year for _, year, *_ in rows],
            PREVENTION_COL: [prev for _, _, prev, _ in rows],
            EXTINCTION_COL: [ext for _, _, _, ext in rows],
        }
    )


def _make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """FIRMS sintético: filas de la forma (ccaa, acq_date)."""
    return pd.DataFrame(
        {
            "ccaa": [ccaa for ccaa, _ in rows],
            "acq_date": [date for _, date in rows],
        }
    )


class TestLoadCcaaBudget:
    def test_real_file_exists(self):
        assert REAL_BUDGET.exists()

    def test_real_file_shape(self):
        df = load_ccaa_budget()
        # 17 CCAA × 6 años (2020-2025)
        assert len(df) == 102
        assert sorted(df[YEAR_COL].unique()) == [2020, 2021, 2022, 2023, 2024, 2025]
        assert df[REGION_COL].nunique() == 17

    def test_real_file_columns(self):
        df = load_ccaa_budget()
        assert list(df.columns) == [
            REGION_COL,
            YEAR_COL,
            "Presupuesto_Total_M€",
            "Presupuesto_Ejecutado_M€",
            PREVENTION_COL,
            EXTINCTION_COL,
            "% Ejecución",
        ]

    def test_real_file_numeric_dtypes(self):
        df = load_ccaa_budget()
        for col in [
            "Presupuesto_Total_M€",
            "Presupuesto_Ejecutado_M€",
            PREVENTION_COL,
            EXTINCTION_COL,
            "% Ejecución",
        ]:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} not numeric"

    def test_real_file_one_row_per_region_year(self):
        df = load_ccaa_budget()
        duplicated = df.duplicated(subset=[REGION_COL, YEAR_COL]).sum()
        assert duplicated == 0

    def test_decimal_comma_parsing(self, tmp_path: Path):
        csv = tmp_path / "budget.csv"
        csv.write_text(
            f"{REGION_COL},{YEAR_COL},{PREVENTION_COL},{EXTINCTION_COL}\n"
            f'Madrid,2023,"12,5","31,1"\n',
            encoding="utf-8",
        )
        df = load_ccaa_budget(csv)
        assert df[PREVENTION_COL].iloc[0] == 12.5
        assert df[EXTINCTION_COL].iloc[0] == 31.1
        assert pd.api.types.is_numeric_dtype(df[PREVENTION_COL])


class TestAddCcaaBudgetSums:
    def test_window_relative_to_row_year(self):
        # prevención = año → la suma distingue qué años entran en la ventana
        budget = _make_budget(
            [
                ("Andalucía", year, float(year), float(year))
                for year in range(2020, 2026)
            ]
        )
        df = _make_df(
            [("Andalucía", "2024-07-15"), ("Andalucía", "2023-01-01")],
        )
        out = add_ccaa_budget_sums(df, n=3, budget=budget)
        # 2024 → 2022+2023+2024; 2023 → 2021+2022+2023
        assert out[SUM_PREVENTION_COL].tolist() == [6069.0, 6066.0]
        assert out[SUM_EXTINCTION_COL].tolist() == [6069.0, 6066.0]

    def test_window_clipped_at_budget_start(self):
        budget = _make_budget(
            [
                ("Andalucía", year, float(year), float(year))
                for year in range(2020, 2026)
            ]
        )
        df = _make_df([("Andalucía", "2023-06-01")])
        out = add_ccaa_budget_sums(df, n=5, budget=budget)
        # ventana 2019..2023, pero 2019 no existe → solo 2020+2021+2022+2023
        assert out[SUM_PREVENTION_COL].iloc[0] == 8086.0

    def test_n_one_is_single_year(self):
        budget = _make_budget([("Madrid", 2024, 10.0, 20.0)])
        df = _make_df([("Madrid", "2024-08-01")])
        out = add_ccaa_budget_sums(df, n=1, budget=budget)
        assert out[SUM_PREVENTION_COL].iloc[0] == 10.0
        assert out[SUM_EXTINCTION_COL].iloc[0] == 20.0

    def test_default_n_from_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            "wildfire.data.ccaa.load_config", lambda: {"ccaa": {"years_window": 2}}
        )
        budget = _make_budget(
            [("Madrid", year, float(year), 0.0) for year in (2023, 2024)]
        )
        df = _make_df([("Madrid", "2024-08-01")])
        out = add_ccaa_budget_sums(df, budget=budget)
        assert out[SUM_PREVENTION_COL].iloc[0] == 4047.0  # 2023 + 2024

    def test_unknown_region_is_nan_not_zero(self):
        budget = _make_budget([("Madrid", 2024, 10.0, 20.0)])
        df = _make_df([("Atlántida", "2024-08-01")])
        out = add_ccaa_budget_sums(df, n=1, budget=budget)
        assert pd.isna(out[SUM_PREVENTION_COL].iloc[0])
        assert pd.isna(out[SUM_EXTINCTION_COL].iloc[0])

    def test_malformed_date_is_nan_not_crash(self):
        budget = _make_budget([("Madrid", 2024, 10.0, 20.0)])
        df = _make_df([("Madrid", "not-a-date"), ("Madrid", None)])
        out = add_ccaa_budget_sums(df, n=1, budget=budget)
        assert pd.isna(out[SUM_PREVENTION_COL].iloc[0])
        assert pd.isna(out[SUM_PREVENTION_COL].iloc[1])
        assert pd.isna(out[SUM_EXTINCTION_COL].iloc[0])
        assert pd.isna(out[SUM_EXTINCTION_COL].iloc[1])

    def test_duplicated_budget_rows_warn_and_keep_first(self):
        budget = _make_budget(
            [
                ("Madrid", 2024, 10.0, 20.0),
                ("Madrid", 2024, 99.0, 99.0),  # duplicada: no debe duplicar la suma
            ]
        )
        df = _make_df([("Madrid", "2024-08-01")])
        with pytest.warns(UserWarning, match="duplicated"):
            out = add_ccaa_budget_sums(df, n=1, budget=budget)
        assert out[SUM_PREVENTION_COL].iloc[0] == 10.0
        assert out[SUM_EXTINCTION_COL].iloc[0] == 20.0

    def test_missing_region_column_raises(self):
        df = pd.DataFrame({"acq_date": ["2024-01-01"]})
        with pytest.raises(KeyError, match="ccaa"):
            add_ccaa_budget_sums(df, n=1, budget=_make_budget([]))

    def test_missing_acq_date_column_raises(self):
        df = pd.DataFrame({"ccaa": ["Madrid"]})
        with pytest.raises(KeyError, match="acq_date"):
            add_ccaa_budget_sums(df, n=1, budget=_make_budget([]))

    def test_invalid_n_raises(self):
        df = _make_df([("Madrid", "2024-08-01")])
        with pytest.raises(ValueError, match="n must be >= 1"):
            add_ccaa_budget_sums(df, n=0, budget=_make_budget([]))

    def test_does_not_mutate_input_and_keeps_index(self):
        budget = _make_budget([("Madrid", 2024, 10.0, 20.0)])
        df = _make_df([("Madrid", "2024-08-01"), ("Madrid", "2023-08-01")])
        df.index = [5, 7]
        original_columns = list(df.columns)
        out = add_ccaa_budget_sums(df, n=1, budget=budget)

        assert list(df.columns) == original_columns  # entrada intacta
        assert out.index.tolist() == [5, 7]
        assert out.columns.tolist() == original_columns + [
            SUM_PREVENTION_COL,
            SUM_EXTINCTION_COL,
        ]

    def test_mixed_years_per_row(self):
        budget = _make_budget(
            [
                ("Madrid", 2023, 1.0, 10.0),
                ("Madrid", 2024, 2.0, 20.0),
                ("Galicia", 2024, 5.0, 50.0),
            ]
        )
        df = _make_df(
            [
                ("Madrid", "2024-01-01"),
                ("Galicia", "2024-03-01"),
                ("Madrid", "2023-01-01"),
            ]
        )
        out = add_ccaa_budget_sums(df, n=1, budget=budget)
        assert out[SUM_PREVENTION_COL].tolist() == [2.0, 5.0, 1.0]
        assert out[SUM_EXTINCTION_COL].tolist() == [20.0, 50.0, 10.0]
