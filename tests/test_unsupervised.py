"""Tests for the unsupervised feature sets (block 03a)."""

import numpy as np
import pandas as pd
import pytest
from wildfire.config import PROJECT_ROOT
from wildfire.processing.pca_weather import pca_feature_columns
from wildfire.processing.unsupervised import (
    CLC_NEIGHBOURS,
    DEFAULT_MIN_CLIFF_DELTA,
    ENTRADA_A,
    ENTRADA_B,
    ENTRADA_C_FIXED,
    ENTRADA_D,
    FREQ_VECINA_PREFIX,
    LOG_COLS,
    PERFILADO,
    add_wind_circular,
    apply_log_transforms,
    assert_feature_sets_clean,
    build_clc_environment,
    clc_environment_columns,
    load_min_cliff_delta,
    resolve_set,
)


def test_set_a_matches_pca_features():
    assert ENTRADA_A == pca_feature_columns()
    assert len(ENTRADA_A) == 14  # 12 base + wd pair


def test_sets_nest():
    assert set(ENTRADA_A) < set(ENTRADA_B)
    assert set(ENTRADA_B) < set(ENTRADA_C_FIXED)
    assert "franja_horaria" in ENTRADA_B
    assert "clc_class" in ENTRADA_C_FIXED


def test_no_satellite_in_inputs():
    # Must not raise.
    assert_feature_sets_clean(ENTRADA_A, ENTRADA_B, ENTRADA_C_FIXED, ENTRADA_D)


def test_guard_raises_on_satellite_leak():
    with pytest.raises(ValueError, match="Satellite columns"):
        assert_feature_sets_clean([*ENTRADA_A, "frp"])
    with pytest.raises(ValueError, match="Satellite columns"):
        assert_feature_sets_clean(["brightness"])


def test_abc_disjoint_from_perfilado():
    # Set D reuses acq_date by design (continuous day for hotspots); A/B/C
    # must not intersect profiling columns.
    for name, columns in [("A", ENTRADA_A), ("B", ENTRADA_B), ("C", ENTRADA_C_FIXED)]:
        assert not (set(columns) & set(PERFILADO)), name


def _toy_clc() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clc_class": [5, 3],
            "clc_class_N": [5, 3],
            "clc_class_S": [5, 7],
            "clc_class_W": [3, 3],
            "clc_class_E": [5, 3],
            "clc_class_NW": [3, 7],
            "clc_class_NE": [5, 3],
            "clc_class_SW": [3, 3],
            "clc_class_SE": [5, 7],
        }
    )


def test_clc_environment_handcomputed():
    env = build_clc_environment(_toy_clc())
    # Row 0: five 5s of eight neighbours, classes {5, 3}.
    assert env["pct_vecinos_igual_centro"].iloc[0] == pytest.approx(5 / 8)
    assert env["n_clases_distintas_vecinos"].iloc[0] == 2
    assert env[f"{FREQ_VECINA_PREFIX}5"].iloc[0] == pytest.approx(5 / 8)
    assert env[f"{FREQ_VECINA_PREFIX}3"].iloc[0] == pytest.approx(3 / 8)
    # Row 1: five 3s of eight neighbours, classes {3, 7}.
    assert env["pct_vecinos_igual_centro"].iloc[1] == pytest.approx(5 / 8)
    assert env["n_clases_distintas_vecinos"].iloc[1] == 2
    assert set(clc_environment_columns(env)) == set(env.columns)


def test_clc_environment_missing_warns():
    with pytest.warns(UserWarning, match="incomplete"):
        env = build_clc_environment(pd.DataFrame({"clc_class": [5]}))
    assert env["pct_vecinos_igual_centro"].isna().all()


def test_log_transforms_values_and_missing():
    df = pd.DataFrame({c: [0.0, 1.0, 3.0] for c in LOG_COLS})
    out = apply_log_transforms(df)
    for c in LOG_COLS:
        assert out[c].tolist() == pytest.approx(np.log1p([0.0, 1.0, 3.0]).tolist())
    with pytest.warns(UserWarning, match="skipped"):
        apply_log_transforms(pd.DataFrame({"precipitation": [1.0]}))


def test_wind_circular_unit_circle():
    df = pd.DataFrame({"wind_direction_10m": [0.0, 90.0, 180.0, 270.0]})
    out = add_wind_circular(df)
    assert ((out[["wd_sin", "wd_cos"]] >= -1.0).all().all())
    assert ((out[["wd_sin", "wd_cos"]] <= 1.0).all().all())
    radius = (out["wd_sin"] ** 2 + out["wd_cos"] ** 2).to_numpy()
    assert radius == pytest.approx(np.ones(4))


def test_min_cliff_delta_default():
    assert DEFAULT_MIN_CLIFF_DELTA == 0.2
    assert load_min_cliff_delta() == pytest.approx(0.2)


def test_resolve_set_warns_and_drops():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.warns(UserWarning, match="absent"):
        out = resolve_set(df, ["a", "zzz"], "TEST")
    assert list(out.columns) == ["a"]


def test_listed_columns_exist_in_lags_index():
    path = PROJECT_ROOT / "data/processed/expanded/firms_spain_final_lags_index.csv"
    if not path.exists():
        pytest.skip("lags_index.csv not present on this checkout")
    cols = set(pd.read_csv(path, nrows=0).columns)
    # wd_sin/wd_cos derive at runtime via AddWindCircular (never stored).
    stored = [c for c in (set(ENTRADA_A) | set(ENTRADA_B)) if c not in ("wd_sin", "wd_cos")]
    for name, columns in [("A/B", stored), ("PERFILADO", PERFILADO)]:
        missing = [c for c in columns if c not in cols]
        assert not missing, f"{name}: {missing}"
    assert set(CLC_NEIGHBOURS) <= cols
