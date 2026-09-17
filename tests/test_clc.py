"""Tests for CLCPlus enrichment utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from wildfire.enrichment.clc_enrichment import (
    CLC_LABELS,
    NEIGHBORHOOD_RINGS,
    _build_tile_index,
    _pixel_value,
    _resolve_neighbor,
    enrich_with_clc,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestClcLabels:
    def test_is_dict(self):
        assert isinstance(CLC_LABELS, dict)

    def test_keys_are_ints(self):
        for key in CLC_LABELS:
            assert isinstance(key, int)

    def test_values_are_strings(self):
        for val in CLC_LABELS.values():
            assert isinstance(val, str)

    def test_expected_classes_present(self):
        expected_codes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 253, 254, 255}
        assert set(CLC_LABELS.keys()) == expected_codes


class TestNeighborhoodRings:
    def test_is_dict(self):
        assert isinstance(NEIGHBORHOOD_RINGS, dict)

    def test_has_rings_1_2_3(self):
        assert set(NEIGHBORHOOD_RINGS.keys()) == {1, 2, 3}

    def test_ring_1_has_4_cardinal_directions(self):
        ring1 = NEIGHBORHOOD_RINGS[1]
        assert len(ring1) == 4
        suffixes = {s for _, _, s in ring1}
        assert suffixes == {"N", "S", "W", "E"}

    def test_ring_2_has_4_diagonals(self):
        ring2 = NEIGHBORHOOD_RINGS[2]
        assert len(ring2) == 4
        suffixes = {s for _, _, s in ring2}
        assert suffixes == {"NW", "NE", "SW", "SE"}

    def test_ring_3_has_4_distance2_cardinals(self):
        ring3 = NEIGHBORHOOD_RINGS[3]
        assert len(ring3) == 4
        suffixes = {s for _, _, s in ring3}
        assert suffixes == {"N2", "S2", "W2", "E2"}

    def test_each_entry_is_3_tuple(self):
        for ring_val in NEIGHBORHOOD_RINGS.values():
            for entry in ring_val:
                assert isinstance(entry, tuple)
                assert len(entry) == 3

    def test_offsets_are_ints(self):
        for ring_val in NEIGHBORHOOD_RINGS.values():
            for dr, dc, suffix in ring_val:
                assert isinstance(dr, int)
                assert isinstance(dc, int)
                assert isinstance(suffix, str)


# ---------------------------------------------------------------------------
# _resolve_neighbor
# ---------------------------------------------------------------------------

class TestResolveNeighbor:
    TILE_H = 10000
    TILE_W = 10000

    def test_no_boundary_crossing(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 5000, -1, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N20"
        assert row == 4999
        assert col == 5000

    def test_move_south(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 5000, 1, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N20"
        assert row == 5001

    def test_move_east(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 5000, 0, 1,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N20"
        assert col == 5001

    def test_cross_north_boundary(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 0, 5000, -1, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N19"
        assert row == 9999

    def test_cross_south_boundary(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 9999, 5000, 1, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N21"
        assert row == 0

    def test_cross_west_boundary(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 0, 0, -1,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E30N20"
        assert col == 9999

    def test_cross_east_boundary(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 9999, 0, 1,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E32N20"
        assert col == 0

    def test_ring3_north_offset(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 1, 5000, -2, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N19"
        assert row == 9999

    def test_ring3_south_offset(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 9998, 5000, 2, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N21"
        assert row == 0

    def test_ring3_west_offset(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 1, 0, -2,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E30N20"
        assert col == 9999

    def test_ring3_east_offset(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 9998, 0, 2,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E32N20"
        assert col == 0

    def test_diagonal_northwest(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 5000, 5000, -1, -1,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E31N20"
        assert row == 4999
        assert col == 4999

    def test_diagonal_northwest_at_origin(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 0, 0, -1, -1,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E30N19"
        assert row == 9999
        assert col == 9999

    def test_custom_tile_size(self):
        tile, row, col = _resolve_neighbor(
            "E31N20", 0, 0, -1, 0,
            tile_height=5000, tile_width=5000,
        )
        assert tile == "E31N19"
        assert row == 4999

    def test_tile_key_parsing(self):
        tile, row, col = _resolve_neighbor(
            "E15N10", 5000, 5000, 0, 0,
            tile_height=self.TILE_H, tile_width=self.TILE_W,
        )
        assert tile == "E15N10"
        assert row == 5000
        assert col == 5000


# ---------------------------------------------------------------------------
# _pixel_value
# ---------------------------------------------------------------------------

class TestPixelValue:
    def test_returns_none_for_none_reader(self):
        result = _pixel_value(None, 0, 0, 100, 100)
        assert result is None

    def test_returns_none_for_negative_row(self):
        mock_src = MagicMock()
        result = _pixel_value(mock_src, -1, 50, 100, 100)
        assert result is None

    def test_returns_none_for_negative_col(self):
        mock_src = MagicMock()
        result = _pixel_value(mock_src, 50, -1, 100, 100)
        assert result is None

    def test_returns_none_for_row_out_of_bounds(self):
        mock_src = MagicMock()
        result = _pixel_value(mock_src, 100, 50, 100, 100)
        assert result is None

    def test_returns_none_for_col_out_of_bounds(self):
        mock_src = MagicMock()
        result = _pixel_value(mock_src, 50, 100, 100, 100)
        assert result is None

    def test_returns_int_for_valid_pixel(self):
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[5.0]])
        mock_src.nodata = None
        result = _pixel_value(mock_src, 10, 20, 100, 100)
        assert result == 5

    def test_returns_none_for_nan_nodata(self):
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[float("nan")]])
        mock_src.nodata = float("nan")
        result = _pixel_value(mock_src, 10, 20, 100, 100)
        assert result is None

    def test_returns_none_for_matching_nodata(self):
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[-9999.0]])
        mock_src.nodata = -9999
        result = _pixel_value(mock_src, 10, 20, 100, 100)
        assert result is None

    def test_returns_none_for_nan_value(self):
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[float("nan")]])
        mock_src.nodata = None
        result = _pixel_value(mock_src, 10, 20, 100, 100)
        assert result is None

    def test_returns_int_truncation(self):
        mock_src = MagicMock()
        mock_src.read.return_value = np.array([[7.9]])
        mock_src.nodata = None
        result = _pixel_value(mock_src, 10, 20, 100, 100)
        assert result == 7
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _build_tile_index
# ---------------------------------------------------------------------------

class TestBuildTileIndex:
    @patch("wildfire.enrichment.clc_enrichment.list_clc_tiles")
    def test_returns_dict(self, mock_list_tiles):
        mock_list_tiles.return_value = []
        result = _build_tile_index()
        assert isinstance(result, dict)

    @patch("wildfire.enrichment.clc_enrichment.list_clc_tiles")
    def test_extracts_tile_key(self, mock_list_tiles, tmp_path):
        tile = tmp_path / "CLMS_CLCPLUS_RAS_S2023_R10m_E31N23_03035_V01_R00.tif"
        tile.touch()
        mock_list_tiles.return_value = [tile]
        result = _build_tile_index()
        assert "E31N23" in result
        assert result["E31N23"] == tile

    @patch("wildfire.enrichment.clc_enrichment.list_clc_tiles")
    def test_multiple_tiles(self, mock_list_tiles, tmp_path):
        tiles = []
        for key in ["E31N20", "E31N21", "E32N20"]:
            t = tmp_path / f"CLMS_CLCPLUS_RAS_S2023_R10m_{key}_03035_V01_R00.tif"
            t.touch()
            tiles.append(t)
        mock_list_tiles.return_value = tiles
        result = _build_tile_index()
        assert len(result) == 3
        assert all(k in result for k in ["E31N20", "E31N21", "E32N20"])

    @patch("wildfire.enrichment.clc_enrichment.list_clc_tiles")
    def test_skips_files_without_tile_key(self, mock_list_tiles, tmp_path):
        good = tmp_path / "CLMS_CLCPLUS_RAS_S2023_R10m_E31N23_03035_V01_R00.tif"
        good.touch()
        bad = tmp_path / "some_other_file.tif"
        bad.touch()
        mock_list_tiles.return_value = [good, bad]
        result = _build_tile_index()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# enrich_with_clc — output schema and uniform surroundings
# ---------------------------------------------------------------------------

class TestEnrichWithClc:
    def _make_firms_df(self, lats, lons):
        return pd.DataFrame({
            "latitude": lats,
            "longitude": lons,
            "acq_date": ["2023-07-01"] * len(lats),
            "acq_time": ["1200"] * len(lats),
            "sensor": ["viirs_snpp"] * len(lats),
        })

    @patch("wildfire.enrichment.clc_enrichment.rasterio")
    @patch("wildfire.enrichment.clc_enrichment._build_tile_index")
    @patch("wildfire.enrichment.clc_enrichment.load_config")
    @patch("wildfire.enrichment.clc_enrichment.Transformer")
    def test_output_has_clc_class_column(self, mock_tf, mock_cfg, mock_idx, mock_rio):
        mock_cfg.return_value = {
            "clcplus": {"neighborhood": {"enabled": False, "rings": []}}
        }
        mock_instance = MagicMock()
        mock_instance.transform.return_value = (
            np.array([3150000.0]),
            np.array([2020000.0]),
        )
        mock_tf.from_crs.return_value = mock_instance
        mock_idx.return_value = {"E31N20": Path("/fake.tif")}

        mock_reader = MagicMock()
        mock_reader.index.return_value = (5000, 5000)
        mock_reader.height = 10000
        mock_reader.width = 10000
        mock_reader.nodata = None
        mock_reader.read.return_value = np.array([[1.0]])
        mock_rio.open.return_value = mock_reader

        df = self._make_firms_df([40.4], [-3.7])
        result = enrich_with_clc(df)
        assert "clc_class" in result.columns

    @patch("wildfire.enrichment.clc_enrichment.rasterio")
    @patch("wildfire.enrichment.clc_enrichment._build_tile_index")
    @patch("wildfire.enrichment.clc_enrichment.load_config")
    @patch("wildfire.enrichment.clc_enrichment.Transformer")
    def test_uniform_surroundings_false_when_no_tile(self, mock_tf, mock_cfg, mock_idx, mock_rio):
        mock_cfg.return_value = {
            "clcplus": {"neighborhood": {"enabled": True, "rings": [1]}}
        }
        mock_instance = MagicMock()
        mock_instance.transform.return_value = (
            np.array([1800000.0]),
            np.array([970000.0]),
        )
        mock_tf.from_crs.return_value = mock_instance
        mock_idx.return_value = {}

        df = self._make_firms_df([28.0], [-15.5])
        result = enrich_with_clc(df)
        assert "clc_uniform_surroundings" in result.columns
        assert not result["clc_uniform_surroundings"].any()

    @patch("wildfire.enrichment.clc_enrichment.rasterio")
    @patch("wildfire.enrichment.clc_enrichment._build_tile_index")
    @patch("wildfire.enrichment.clc_enrichment.load_config")
    @patch("wildfire.enrichment.clc_enrichment.Transformer")
    def test_does_not_modify_original_df(self, mock_tf, mock_cfg, mock_idx, mock_rio):
        mock_cfg.return_value = {
            "clcplus": {"neighborhood": {"enabled": False, "rings": []}}
        }
        mock_instance = MagicMock()
        mock_instance.transform.return_value = (
            np.array([3150000.0]),
            np.array([2020000.0]),
        )
        mock_tf.from_crs.return_value = mock_instance
        mock_idx.return_value = {}

        df = self._make_firms_df([40.4], [-3.7])
        original_cols = list(df.columns)
        enrich_with_clc(df)
        assert list(df.columns) == original_cols

    @patch("wildfire.enrichment.clc_enrichment.rasterio")
    @patch("wildfire.enrichment.clc_enrichment._build_tile_index")
    @patch("wildfire.enrichment.clc_enrichment.load_config")
    @patch("wildfire.enrichment.clc_enrichment.Transformer")
    def test_neighborhood_columns_added(self, mock_tf, mock_cfg, mock_idx, mock_rio):
        mock_cfg.return_value = {
            "clcplus": {"neighborhood": {"enabled": True, "rings": [1]}}
        }
        mock_instance = MagicMock()
        mock_instance.transform.return_value = (
            np.array([3150000.0]),
            np.array([2020000.0]),
        )
        mock_tf.from_crs.return_value = mock_instance
        mock_idx.return_value = {"E31N20": Path("/fake.tif")}

        mock_reader = MagicMock()
        mock_reader.index.return_value = (5000, 5000)
        mock_reader.height = 10000
        mock_reader.width = 10000
        mock_reader.nodata = None
        mock_reader.read.return_value = np.array([[1.0]])
        mock_rio.open.return_value = mock_reader

        df = self._make_firms_df([40.4], [-3.7])
        result = enrich_with_clc(df)
        for suffix in ["N", "S", "W", "E"]:
            assert f"clc_class_{suffix}" in result.columns


# ---------------------------------------------------------------------------
# Enrichment __init__ exports
# ---------------------------------------------------------------------------

class TestEnrichmentExports:
    def test_clc_labels_exported(self):
        from wildfire.enrichment import CLC_LABELS as exported
        assert exported is CLC_LABELS

    def test_neighborhood_rings_exported(self):
        from wildfire.enrichment import NEIGHBORHOOD_RINGS as exported
        assert exported is NEIGHBORHOOD_RINGS

    def test_load_clc_classes_not_exported(self):
        import wildfire.enrichment as enrichment
        assert not hasattr(enrichment, "load_clc_classes")
        assert not hasattr(enrichment, "_load_clc_classes")
