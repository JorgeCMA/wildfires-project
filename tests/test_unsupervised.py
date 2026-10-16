"""Tests for the unsupervised feature sets (block 03a)."""

import numpy as np
import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT
from wildfire.processing.pca_weather import pca_feature_columns
from wildfire.processing.unsupervised import (
    CLC_NEIGHBOURS,
    CLUSTERING_SEED,
    DEFAULT_MIN_CLIFF_DELTA,
    ENTRADA_A,
    ENTRADA_B,
    ENTRADA_C_FIXED,
    ENTRADA_D,
    FRANJA_ORDER,
    FREQ_VECINA_PREFIX,
    ID_COL,
    K_RANGE,
    LOG_COLS,
    NOMINAL_EXCLUDED,
    PERFILADO,
    WARD_SAMPLE_N,
    add_wind_circular,
    apply_log_transforms,
    assert_feature_sets_clean,
    attach_id,
    build_clc_environment,
    check_ids,
    clc_environment_columns,
    cliffs_delta,
    distance_columns,
    encode_franja_horaria,
    fit_gmm_range,
    fit_kmeans_range,
    gmm_bic_table,
    load_min_cliff_delta,
    max_pairwise_cliffs_delta,
    negative_control_partitions,
    resolve_set,
    sample_frame,
    ward_linkage,
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
    # Anclado en 03e contra baselines reales (CCAA = 0,699 -> listón 0,70).
    assert load_min_cliff_delta() == pytest.approx(0.70)


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

# ---------------------------------------------------------------------------
# Block 03b helpers.
# ---------------------------------------------------------------------------


def _blobs(n=60, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(-3.0, 0.5, size=(n, 2))
    b = rng.normal(3.0, 0.5, size=(n, 2))
    return pd.DataFrame(np.vstack([a, b]), columns=["x", "y"])


def test_k_range_and_seed():
    assert K_RANGE == list(range(2, 13))
    assert CLUSTERING_SEED == 42
    assert WARD_SAMPLE_N == 5_000


def test_franja_encoding_order():
    df = pd.DataFrame({"franja_horaria": FRANJA_ORDER})
    out = encode_franja_horaria(df)
    assert out["franja_horaria"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_franja_unknown_warns_to_nan():
    with pytest.warns(UserWarning, match="Unknown"):
        out = encode_franja_horaria(pd.DataFrame({"franja_horaria": ["Madrugada", "Siesta"]}))
    assert out["franja_horaria"].iloc[0] == 0.0
    assert np.isnan(out["franja_horaria"].iloc[1])


def test_distance_columns_drop_nominal():
    assert distance_columns([*ENTRADA_C_FIXED, "franja_horaria"]) == [
        c for c in [*ENTRADA_C_FIXED, "franja_horaria"] if c not in NOMINAL_EXCLUDED
    ]
    assert "clc_class" not in distance_columns(ENTRADA_C_FIXED)


def test_resolve_set_dedupes():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = resolve_set(df, ["a", "b", "a"], "DUP")
    assert list(out.columns) == ["a", "b"]


def test_kmeans_range_deterministic_shapes():
    X = _blobs()
    labels1, m1 = fit_kmeans_range(X, k_range=[2, 3], seed=42)
    labels2, m2 = fit_kmeans_range(X, k_range=[2, 3], seed=42)
    assert set(labels1) == {2, 3}
    assert all(len(v) == len(X) for v in labels1.values())
    assert (labels1[2] == labels2[2]).all()
    assert m1.equals(m2)
    assert list(m1.columns) == ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]
    assert m1.loc[3, "inertia"] < m1.loc[2, "inertia"]  # blobs separate cleanly


def test_gmm_range_proba_and_bic():
    X = _blobs()
    res = fit_gmm_range(X, k_range=[2], covariance_types=("full", "diag"))
    for (k, cov), r in res.items():
        assert r["proba"].shape == (len(X), k)
        assert r["proba"].sum(axis=1) == pytest.approx(np.ones(len(X)))
        assert np.isfinite(r["bic"])
        assert len(r["labels"]) == len(X)
    table = gmm_bic_table(res)
    assert table.shape == (1, 2)


def test_ward_linkage_shape_and_sample():
    X = _blobs(n=40)
    Z = ward_linkage(X)
    assert Z.shape == (len(X) - 1, 4)
    s1 = sample_frame(X, 25, seed=42)
    s2 = sample_frame(X, 25, seed=42)
    assert s1.equals(s2)


def test_cliffs_delta_handcomputed():
    assert cliffs_delta([3, 4, 5], [1, 2, 1]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert cliffs_delta([1, 1], [2, 3]) == pytest.approx(-1.0)


def test_max_pairwise_and_controls():
    values = np.array([10.0, 11.0, 1.0, 2.0, 5.0, 6.0])
    labels = np.array([0, 0, 1, 1, 2, 2])
    delta, pair = max_pairwise_cliffs_delta(values, labels)
    assert abs(delta) == pytest.approx(1.0)
    assert set(pair) == {0, 1}
    perf = pd.DataFrame({"month": [1, 1, 2, 2], "ccaa": ["A", "B", "A", "B"]})
    controls = negative_control_partitions(perf)
    assert controls["month"].tolist() == [0, 0, 1, 1]
    assert controls["ccaa"].tolist() == [0, 1, 0, 1]

# ---------------------------------------------------------------------------
# Block 03c helpers.
# ---------------------------------------------------------------------------


def _with_outliers(n=200, seed=1):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n, 3))
    out = np.array([[10.0, 10.0, 10.0], [-9.0, 9.0, -9.0]])
    return pd.DataFrame(np.vstack([base, out]), columns=["x", "y", "z"])


def test_anomaly_constants():
    from wildfire.processing.unsupervised import ANOMALY_PCT, CONSENSUS_MIN_VOTES

    assert ANOMALY_PCT == 2.0
    assert CONSENSUS_MIN_VOTES == 2


def test_detectors_rank_outliers_last():
    from wildfire.processing.unsupervised import (
        fit_isolation_forest,
        fit_lof,
        robust_mahalanobis,
    )

    X = _with_outliers()
    for scores in (fit_isolation_forest(X), fit_lof(X), robust_mahalanobis(X)):
        assert len(scores) == len(X)
        assert np.all(np.isfinite(scores))
        order = np.argsort(scores)
        assert set(order[-2:].tolist()) == {len(X) - 2, len(X) - 1}


def test_detectors_deterministic():
    from wildfire.processing.unsupervised import (
        fit_isolation_forest,
        robust_mahalanobis,
    )

    X = _with_outliers()
    assert (fit_isolation_forest(X) == fit_isolation_forest(X)).all()
    assert (robust_mahalanobis(X) == robust_mahalanobis(X)).all()


def test_percentile_flag_rate_and_bounds():
    from wildfire.processing.unsupervised import percentile_flag

    s = np.arange(1000, dtype=float)
    flag = percentile_flag(s, pct=2.0)
    assert flag.sum() == 20
    assert flag[-1] and not flag[0]
    with pytest.raises(ValueError, match="pct"):
        percentile_flag(s, pct=0.0)


def test_consensus_flag_votes():
    from wildfire.processing.unsupervised import consensus_flag

    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    c = np.array([False, False, False, True])
    assert consensus_flag(a, b, c).tolist() == [True, False, False, False]
    assert consensus_flag(a, b, c, min_votes=1).tolist() == [True, True, True, True]


def test_imputed_blh_mask():
    from wildfire.processing.unsupervised import imputed_blh_mask

    df = pd.DataFrame({"boundary_layer_height": [100.0, 577.5, 577.5, 900.0]})
    # Median of [100, 577.5, 577.5, 900] is 577.5 -> rows 1, 2 flagged.
    assert imputed_blh_mask(df).tolist() == [False, True, True, False]
    with pytest.warns(UserWarning, match="absent"):
        out = imputed_blh_mask(pd.DataFrame({"x": [1]}))
    assert not out.any()


def test_weather_pc_scores_shape_and_index():
    from wildfire.processing.pca_weather import pca_input_columns
    from wildfire.processing.unsupervised import weather_pc_scores

    path = PROJECT_ROOT / "data/processed/expanded/firms_spain_final_lags_index.csv"
    if not path.exists():
        pytest.skip("lags_index.csv not present on this checkout")
    df = pd.read_csv(path, usecols=pca_input_columns())
    scores = weather_pc_scores(df)
    assert list(scores.columns) == [f"PC{i + 1}" for i in range(scores.shape[1])]
    assert scores.shape[0] == len(df)
    assert (scores.index == df.index).all()


def test_attach_id_first_column_sequence():
    df = pd.DataFrame({"x": [10, 20, 30]})
    out = attach_id(df)
    assert next(iter(out.columns)) == ID_COL
    assert out[ID_COL].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="already present"):
        attach_id(out)


def test_check_ids_equal_and_mismatch():
    a = attach_id(pd.DataFrame({"x": [1, 2]}))
    b = attach_id(pd.DataFrame({"y": [3, 4]}))
    assert check_ids(a, b).tolist() == [0, 1]
    shuffled = b.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="mismatch"):
        check_ids(a, shuffled)
    with pytest.raises(KeyError, match=ID_COL):
        check_ids(a, pd.DataFrame({"y": [3, 4]}))
    with pytest.raises(ValueError, match="at least one"):
        check_ids()


def test_distance_columns_drops_key():
    assert ID_COL not in distance_columns(["PC1", "clc_class", ID_COL])
    assert distance_columns(["PC1", "PC2"]) == ["PC1", "PC2"]
