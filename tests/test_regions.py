"""Tests para la asignación de CCAA (`wildfire.geo.regions`)."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from wildfire.config import PROJECT_ROOT
from wildfire.data.ccaa import (
    REGION_COL,
    SUM_EXTINCTION_COL,
    SUM_PREVENTION_COL,
    add_ccaa_budget_sums,
    load_ccaa_budget,
)
from wildfire.geo.regions import (
    CCAA_COL,
    REGION_NAME_MAP,
    assign_ccaa,
    load_boundaries,
)

# Datos reales (tracked / generados por el pipeline).
REAL_GEOJSON = PROJECT_ROOT / "data" / "raw" / "ccaa" / "spain-communities.geojson"
REAL_CLC_CSV = (
    PROJECT_ROOT / "data" / "processed" / "enriched" / "firms_spain_weather_clc.csv"
)

# Nombres que existen en el GeoJSON pero NO en el presupuesto.
UNBUDGETED = {"Ceuta", "Melilla"}


def _make_boundaries() -> gpd.GeoDataFrame:
    """Dos polígonos solapados (intersección lon [-3.6, -3.4], lat [40, 41])."""
    return gpd.GeoDataFrame(
        {"name": ["RegionA", "RegionB"]},
        geometry=[
            box(-4.0, 40.0, -3.4, 41.0),
            box(-3.6, 40.0, -3.0, 41.0),
        ],
        crs="EPSG:4326",
    )


def _make_df(rows: list[tuple[float, float]]) -> pd.DataFrame:
    """FIRMS sintético: filas de la forma (longitude, latitude)."""
    return pd.DataFrame(
        {
            "longitude": [lon for lon, _ in rows],
            "latitude": [lat for _, lat in rows],
        }
    )


class TestRegionNameMap:
    def test_has_17_entries(self):
        assert len(REGION_NAME_MAP) == 17

    def test_values_are_exactly_the_budget_regions(self):
        budget = load_ccaa_budget()
        assert set(REGION_NAME_MAP.values()) == set(budget[REGION_COL].unique())

    def test_keys_are_geojson_names_without_ceuta_melilla(self):
        boundaries = load_boundaries(REAL_GEOJSON)
        names = set(boundaries["name"])
        assert set(REGION_NAME_MAP) | UNBUDGETED == names
        assert not (set(REGION_NAME_MAP) & UNBUDGETED)


class TestLoadBoundaries:
    def test_real_file_shape(self):
        boundaries = load_boundaries()
        assert len(boundaries) == 19  # 17 CCAA + Ceuta + Melilla
        assert boundaries.crs is not None
        assert boundaries.crs.to_epsg() == 4326

    def test_real_file_geometries_valid_after_load(self):
        boundaries = load_boundaries()
        assert boundaries.geometry.is_valid.all()

    def test_real_file_has_name_field(self):
        boundaries = load_boundaries()
        assert "name" in boundaries.columns
        assert boundaries["name"].nunique() == 19

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="boundaries not found"):
            load_boundaries(tmp_path / "nope.geojson")

    def test_absolute_path_wins_over_config(self, tmp_path: Path):
        gdf = _make_boundaries()
        path = tmp_path / "synthetic.geojson"
        gdf.to_file(path, driver="GeoJSON")
        loaded = load_boundaries(path)
        assert len(loaded) == 2
        assert list(loaded["name"]) == ["RegionA", "RegionB"]


class TestAssignCcaaSynthetic:
    def test_within_assignment(self):
        df = _make_df([(-3.7, 40.5), (-3.2, 40.5)])  # A, B
        out = assign_ccaa(df, boundaries=_make_boundaries())
        assert out[CCAA_COL].tolist() == ["RegionA", "RegionB"]

    def test_overlapping_border_point_deduped(self):
        df = _make_df([(-3.5, 40.5)])  # dentro del solape → 2 polígonos
        out = assign_ccaa(df, boundaries=_make_boundaries())
        assert len(out) == 1
        assert out.index.is_unique
        assert out[CCAA_COL].iloc[0] in ("RegionA", "RegionB")

    def test_unmatched_beyond_cap_is_nan(self):
        df = _make_df([(-5.0, 40.5)])  # ~84 km al oeste de RegionA
        out = assign_ccaa(df, boundaries=_make_boundaries(), max_distance_m=2000)
        assert pd.isna(out[CCAA_COL].iloc[0])

    def test_nearest_fallback_within_cap(self):
        df = _make_df([(-4.004, 40.5)])  # ~0.3 km al oeste de RegionA
        out = assign_ccaa(df, boundaries=_make_boundaries(), max_distance_m=2000)
        assert out[CCAA_COL].iloc[0] == "RegionA"

    def test_nearest_disabled_with_zero_cap(self):
        df = _make_df([(-4.004, 40.5)])
        out = assign_ccaa(df, boundaries=_make_boundaries(), max_distance_m=0)
        assert pd.isna(out[CCAA_COL].iloc[0])

    def test_appends_ccaa_last_and_keeps_index(self):
        df = _make_df([(-3.7, 40.5), (-3.2, 40.5)])
        df.index = [5, 7]
        original_columns = list(df.columns)
        out = assign_ccaa(df, boundaries=_make_boundaries())

        assert list(df.columns) == original_columns  # entrada intacta
        assert out.index.tolist() == [5, 7]
        assert out.columns.tolist() == original_columns + [CCAA_COL]

    def test_does_not_mutate_input(self):
        df = _make_df([(-3.7, 40.5)])
        assign_ccaa(df, boundaries=_make_boundaries())
        assert CCAA_COL not in df.columns

    def test_row_count_and_order_preserved(self):
        df = _make_df(
            [
                (-3.7, 40.5),
                (-5.0, 40.5),  # sin región (fuera del tope)
                (-3.5, 40.5),  # frontera → 2 candidatos
                (-3.2, 40.5),
            ]
        )
        out = assign_ccaa(df, boundaries=_make_boundaries(), max_distance_m=2000)
        assert len(out) == len(df)
        assert out.index.tolist() == df.index.tolist()

    def test_missing_latitude_raises(self):
        df = pd.DataFrame({"longitude": [-3.7]})
        with pytest.raises(KeyError, match="latitude"):
            assign_ccaa(df, boundaries=_make_boundaries())

    def test_missing_longitude_raises(self):
        df = pd.DataFrame({"latitude": [40.5]})
        with pytest.raises(KeyError, match="longitude"):
            assign_ccaa(df, boundaries=_make_boundaries())

    def test_missing_name_field_raises(self):
        df = _make_df([(-3.7, 40.5)])
        boundaries = _make_boundaries().rename(columns={"name": "otro"})
        with pytest.raises(KeyError, match="otro"):
            assign_ccaa(df, boundaries=boundaries)

    def test_empty_df_returns_empty_with_ccaa(self):
        out = assign_ccaa(_make_df([]), boundaries=_make_boundaries())
        assert len(out) == 0
        assert list(out.columns) == ["longitude", "latitude", CCAA_COL]

    def test_nan_coords_shortcircuit_to_nan(self):
        df = pd.DataFrame(
            {"longitude": [float("nan"), -3.7], "latitude": [40.5, float("nan")]}
        )
        out = assign_ccaa(df, boundaries=_make_boundaries())
        assert out[CCAA_COL].isna().all()

    def test_duplicate_input_index_not_collapsed(self):
        df = _make_df([(-3.7, 40.5), (-3.2, 40.5)])
        df.index = [7, 7]
        with pytest.warns(UserWarning, match="without budget mapping"):
            out = assign_ccaa(df, boundaries=_make_boundaries())
        assert len(out) == 2
        assert out.index.tolist() == [7, 7]
        assert out[CCAA_COL].tolist() == ["RegionA", "RegionB"]

    def test_negative_max_distance_raises(self):
        df = _make_df([(-3.7, 40.5)])
        with pytest.raises(ValueError, match="max_distance_m"):
            assign_ccaa(df, boundaries=_make_boundaries(), max_distance_m=-1)

    def test_unmapped_names_warn(self):
        df = _make_df([(-3.7, 40.5)])
        with pytest.warns(UserWarning, match="without budget mapping"):
            out = assign_ccaa(df, boundaries=_make_boundaries())
        assert out[CCAA_COL].iloc[0] == "RegionA"  # pasa sin traducir


class TestAssignCcaaRealData:
    """Sobre firms_spain_weather_clc.csv (32.500 filas) + GeoJSON real."""

    def test_real_inputs_exist(self):
        assert REAL_GEOJSON.exists()
        assert REAL_CLC_CSV.exists()

    def test_smoke_on_real_data(self):
        raw = pd.read_csv(REAL_CLC_CSV, usecols=["latitude", "longitude"])
        out = assign_ccaa(raw)

        assert len(out) == len(raw)  # ninguna fila se pierde
        # 99 puntos de costa/frontera usan el respaldo ≤2 km → casi todo
        # casado; el tope actual cubre el 100 % (0 NaN), pero el umbral
        # protege contra refrescos de datos.
        assert out[CCAA_COL].notna().mean() > 0.99

        allowed = set(REGION_NAME_MAP.values()) | UNBUDGETED
        assert set(out[CCAA_COL].dropna().unique()) <= allowed

    def test_feeds_budget_sums(self):
        df = pd.DataFrame(
            {
                "longitude": [-3.7038],  # Madrid capital
                "latitude": [40.4168],
                "acq_date": ["2024-07-15"],
            }
        )
        out = add_ccaa_budget_sums(assign_ccaa(df))
        assert out[CCAA_COL].iloc[0] == "Madrid"
        # ventana 2022+2023+2024 de Madrid → importe real, no NaN
        assert out[SUM_PREVENTION_COL].notna().iloc[0]
        assert out[SUM_EXTINCTION_COL].notna().iloc[0]
