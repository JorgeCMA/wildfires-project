"""Tests for the synthetic background generator."""

import pandas as pd

from wildfire.data.background import (
    BACKGROUND_TYPE,
    COUNTRY,
    SATELLITAL_NAN_COLS,
    generate_background,
)
from wildfire.data.io import load_csv

MERGED = "data/processed/merged/firms_spain_merged.csv"


def _small(n=400, seed=7, tmp_path=None):
    out = tmp_path / f"bg_{seed}.csv" if tmp_path is not None else None
    kwargs = {} if out is None else {"output": out}
    return generate_background(n=n, seed=seed, **kwargs)


def test_schema_matches_merged(tmp_path):
    df = _small(tmp_path=tmp_path)
    assert list(df.columns) == list(load_csv(MERGED, nrows=0).columns)


def test_deterministic_same_seed(tmp_path):
    a = _small(seed=11, tmp_path=tmp_path)
    b = generate_background(n=400, seed=11, output=tmp_path / "bg_11b.csv")
    pd.testing.assert_frame_equal(a, b)


def test_structural_columns(tmp_path):
    df = _small(tmp_path=tmp_path)
    assert (df["country"] == COUNTRY).all()
    assert (df["type"] == BACKGROUND_TYPE).all()
    assert (df["year"].astype(str) == df["acq_date"].str[:4]).all()
    assert df["acq_date"].between("2023-01-01", "2024-12-31").all()
    hhmm = df["acq_time"].astype(int)
    assert ((hhmm // 100 <= 23) & (hhmm % 100 <= 59)).all()
    assert set(df["daynight"].unique()) <= {"D", "N"}
    for col in SATELLITAL_NAN_COLS:
        assert df[col].isna().all(), col


def test_points_on_spanish_land(tmp_path):
    from shapely import contains_xy

    from wildfire.geo.regions import load_boundaries

    df = _small(tmp_path=tmp_path)
    land = load_boundaries().geometry.union_all()
    assert bool(
        contains_xy(
            land, df["longitude"].to_numpy(), df["latitude"].to_numpy()
        ).all()
    )
