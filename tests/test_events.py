"""Tests for the spatio-temporal fire events (block 03d)."""

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.events import (
    DAY_KM_SENSITIVITY,
    DEFAULT_DAY_KM,
    DEFAULT_MIN_SAMPLES,
    EVENT_INPUT_COLS,
    EVENT_MATRIX_COLS,
    EVENT_SUMMARY_BASE,
    NOISE_LABEL,
    anomaly_density_per_event,
    build_event_matrix,
    cluster_span_per_event,
    fit_dbscan_events,
    k_distance_curve,
    summarize_events,
)


def _toy_df(n=6, **overrides):
    base = {
        "latitude": [40.0] * n,
        "longitude": [-4.0] * n,
        "acq_date": pd.date_range("2023-07-01", periods=n, freq="D").astype(str),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_defaults_and_sensitivity_grid():
    assert DEFAULT_DAY_KM in DAY_KM_SENSITIVITY
    assert DEFAULT_MIN_SAMPLES >= 1
    # input columns suffice even with extra columns present
    df = _toy_df()
    df["extra"] = 1.0
    m = build_event_matrix(df[EVENT_INPUT_COLS + ["extra"]])
    assert list(m.columns) == EVENT_MATRIX_COLS
    # default min_samples works without passing it
    labels = fit_dbscan_events(np.zeros((4, 3)), eps=0.1)
    assert len(labels) == 4


def test_matrix_columns_and_day_scaling():
    m = build_event_matrix(_toy_df())
    assert list(m.columns) == EVENT_MATRIX_COLS
    assert m["day"].tolist() == [0, 1, 2, 3, 4, 5]
    assert np.allclose(m["day_scaled"], m["day"] * DEFAULT_DAY_KM)
    m2 = build_event_matrix(_toy_df(), day_km=5.0)
    assert np.allclose(m2["day_scaled"], m2["day"] * 5.0)


def test_matrix_preserves_index():
    df = _toy_df()
    df.index = [10, 20, 30, 40, 50, 60]
    assert build_event_matrix(df).index.tolist() == [10, 20, 30, 40, 50, 60]


def test_projection_scale_sanity():
    # 1 degree of latitude is ~111 km anywhere on Earth.
    df = _toy_df(n=2, latitude=[40.0, 41.0])
    m = build_event_matrix(df)
    assert 100.0 < abs(m["y_km"].iloc[1] - m["y_km"].iloc[0]) < 125.0


def test_matrix_guards():
    df = _toy_df().drop(columns=["latitude"])
    with pytest.raises(KeyError):
        build_event_matrix(df)
    with pytest.raises(ValueError):
        build_event_matrix(_toy_df().iloc[0:0])
    with pytest.raises(ValueError):
        build_event_matrix(_toy_df(n=3, acq_date=["2023-07-01", "nope", "2023-07-03"]))
    with pytest.raises(ValueError):
        build_event_matrix(_toy_df(), day_km=0)


def test_k_distance_curve_sorted_and_guards():
    X = pd.DataFrame({"x_km": [0.0, 1.0, 2.0, 10.0], "y_km": [0.0] * 4, "day_scaled": [0.0] * 4})
    curve = k_distance_curve(X, k=1)
    assert len(curve) == 4
    assert bool(np.all(np.diff(curve) >= 0))
    with pytest.raises(ValueError):
        k_distance_curve(X, k=0)
    with pytest.raises(ValueError):
        k_distance_curve(X.iloc[:1], k=1)


def test_dbscan_two_blobs_plus_noise():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [50.0, 50.0, 50.0],
            [50.5, 50.0, 50.0],
            [50.0, 50.5, 50.0],
            [500.0, 500.0, 500.0],
        ]
    )
    labels = fit_dbscan_events(X, eps=2.0, min_samples=3)
    assert set(labels) == {0, 1, NOISE_LABEL}
    assert (labels == NOISE_LABEL).sum() == 1


def test_dbscan_guards():
    X = np.zeros((5, 3))
    with pytest.raises(ValueError):
        fit_dbscan_events(X, eps=0)
    with pytest.raises(ValueError):
        fit_dbscan_events(X, eps=1.0, min_samples=0)
    with pytest.raises(ValueError):
        fit_dbscan_events(np.zeros((0, 3)), eps=1.0)


def test_summarize_events_math():
    frame = pd.DataFrame(
        {
            "x_km": [0.0, 2.0, 100.0, 101.0],
            "y_km": [0.0, 0.0, 5.0, 5.0],
            "day": [0, 2, 0, 0],
            "ccaa": ["A", "A", "B", "C"],
            "frp": [5.0, 9.0, 3.0, 4.0],
            "PC1": [1.0, 3.0, 0.0, 0.0],
        }
    )
    labels = np.array([0, 0, 1, 1])
    s = summarize_events(frame, labels, pc_cols=["PC1"])
    assert list(s.columns) == [*EVENT_SUMMARY_BASE, "PC1_mean"]
    ev0 = s[s["event_id"] == 0].iloc[0]
    assert ev0["n"] == 2
    assert ev0["duration_days"] == 3
    assert ev0["x_span_km"] == 2.0
    assert ev0["ccaa_top"] == "A" and ev0["ccaa_n"] == 1
    assert ev0["frp_max"] == 9.0
    assert ev0["PC1_mean"] == 2.0
    ev1 = s[s["event_id"] == 1].iloc[0]
    assert ev1["ccaa_n"] == 2


def test_summarize_excludes_noise_and_empty_schema():
    frame = pd.DataFrame(
        {"x_km": [0.0], "y_km": [0.0], "day": [0], "ccaa": ["A"], "frp": [1.0]}
    )
    empty = summarize_events(frame, np.array([NOISE_LABEL]), pc_cols=[])
    assert len(empty) == 0
    assert list(empty.columns) == EVENT_SUMMARY_BASE
    with pytest.raises(KeyError):
        summarize_events(frame.drop(columns=["frp"]), np.array([0]))
    with pytest.raises(KeyError):
        summarize_events(frame, np.array([0]), pc_cols=["PC9"])
    with pytest.raises(ValueError):
        summarize_events(frame, np.array([0, 1]))


def test_cluster_span_counts_regimes():
    span = cluster_span_per_event(
        pd.Series([0, 0, 0, 1, 1]), pd.Series([2, 2, 5, 5, 5])
    )
    assert span.set_index("event_id")["n_clusters"].to_dict() == {0: 2, 1: 1}
    assert span.set_index("event_id")["cluster_ids"].to_dict() == {0: "2;5", 1: "5"}
    with pytest.raises(ValueError):
        cluster_span_per_event([0, 1], [0])


def test_anomaly_density_fractions():
    dens = anomaly_density_per_event(
        pd.Series([0, 0, 0, 0, 1, 1]), pd.Series([True, False, False, False, True, True])
    )
    row = dens.set_index("event_id")
    assert row.loc[0, "n"] == 4 and row.loc[0, "n_flagged"] == 1
    assert row.loc[0, "frac_flagged"] == pytest.approx(0.25)
    assert row.loc[1, "frac_flagged"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        anomaly_density_per_event([0], [True, False])
