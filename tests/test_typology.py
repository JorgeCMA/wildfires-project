"""Tests for the typology validation (block 03e)."""

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.typology import (
    GMM_OVERLAP_MAX,
    GMM_OVERLAP_POSTERIOR,
    NMI_TRIVIALITY_MAX,
    PREPROCESSING_ARI_MIN,
    SPATIAL_CORR_MIN,
    STABILITY_ARI_MIN,
    anchor_min_delta,
    cross_year_profiles,
    fit_labels,
    gmm_overlap_mass,
    leave_one_group_out,
    noncoincidence_ari,
    nontriviality_nmi,
    pairwise_ari,
    resample_stability,
    severity_separation,
    validation_verdict,
)


def _blobs(n_per=60, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0.0, scale=0.3, size=(n_per, 2))
    b = rng.normal(loc=5.0, scale=0.3, size=(n_per, 2))
    return np.vstack([a, b])


def test_fit_labels_deterministic_and_guards():
    X = _blobs()
    assert (fit_labels(X, 2) == fit_labels(X, 2)).all()
    with pytest.raises(ValueError):
        fit_labels(X, 1)


def test_resample_stability_clean_blobs_pass():
    out = resample_stability(_blobs(), k=2, n_resamples=10, frac=0.8)
    assert out["mean_ari"] > STABILITY_ARI_MIN
    assert out["min_ari"] <= out["mean_ari"]
    assert len(out["aris"]) == 10
    with pytest.raises(ValueError):
        resample_stability(np.zeros((0, 2)), k=2)
    with pytest.raises(ValueError):
        resample_stability(_blobs(), k=2, frac=0.0)


def test_leave_one_group_out_structure():
    X = _blobs(n_per=40)
    groups = pd.Series(["g1"] * 40 + ["g2"] * 40)
    table = leave_one_group_out(X, groups, k=2)
    assert set(table["group"]) == {"g1", "g2"}
    assert (table["n_heldout"] == 40).all()
    assert (table["groups_found"] == 2).all()
    assert (table["profile_corr"] > 0.9).all()


def test_profile_matching_ignores_relabelling():
    a = pd.DataFrame({"x": [0.0, 5.0], "y": [0.0, 5.0]})
    b = pd.DataFrame({"x": [5.0, 0.0], "y": [5.0, 0.0]})  # same groups, swapped rows
    from wildfire.processing.typology import _match_profile_corr

    assert _match_profile_corr(a, b) == pytest.approx(1.0)
    c = pd.DataFrame({"x": [0.0, 5.0, 9.0], "y": [0.0, 5.0, 9.0]})
    assert np.isnan(_match_profile_corr(a, c))


def test_cross_year_profiles_replicate():
    rng = np.random.default_rng(1)
    X = np.vstack(
        [rng.normal(0.0, 0.3, (50, 2)), rng.normal(5.0, 0.3, (50, 2))] * 2
    )
    year = pd.Series([2023] * 100 + [2024] * 100)
    out = cross_year_profiles(X, year, k=2)
    assert out["profile_corr"] > 0.9
    with pytest.raises(ValueError):
        cross_year_profiles(X, pd.Series([2023] * 200), k=2)


def test_pairwise_ari_identity_and_guards():
    lab = np.array([0, 0, 1, 1, 2, 2])
    table = pairwise_ari({"a": lab, "b": lab.copy(), "c": np.array([1, 1, 0, 0, 2, 2])})
    assert table.loc["a", "b"] == pytest.approx(1.0)
    assert table.loc["a", "c"] == pytest.approx(1.0)  # relabelled identical
    assert (table.to_numpy() >= 0).all()
    with pytest.raises(ValueError):
        pairwise_ari({"only": lab})
    with pytest.raises(ValueError):
        pairwise_ari({"a": lab, "b": lab[:3]})
    assert PREPROCESSING_ARI_MIN == 0.6


def test_severity_separation_separated_groups_pass():
    rng = np.random.default_rng(2)
    lab = np.array([0] * 50 + [1] * 50)
    frp = np.concatenate([rng.normal(5, 1, 50), rng.normal(50, 5, 50)])
    out = severity_separation(lab, frp)
    assert out["kruskal_p"] < 1e-6
    assert out["passes"] is True
    with pytest.raises(ValueError):
        severity_separation(lab, frp[:10])


def test_noncoincidence_calendar_is_calendar():
    month = np.array([1] * 30 + [2] * 30)
    ccaa = np.array(["a"] * 20 + ["b"] * 40)
    out = noncoincidence_ari(month, month, ccaa)
    assert out["ari_vs_month"] == pytest.approx(1.0)
    assert out["ari_vs_ccaa"] < 1.0


def test_nontriviality_flags_single_variable_groups():
    month = np.array([1] * 30 + [2] * 30)
    other = np.array([0, 1] * 30)
    out = nontriviality_nmi(month, month=month, other=other)
    assert out["month"] == pytest.approx(1.0)
    assert out["passes"] is False
    assert NMI_TRIVIALITY_MAX == 0.5


def test_gmm_overlap_mass_confident_vs_flat():
    confident = np.array([[0.95, 0.05]] * 90 + [[0.5, 0.5]] * 10)
    out = gmm_overlap_mass(confident)
    assert out["frac_below"] == pytest.approx(0.1)
    assert out["passes"] is True
    flat = np.full((20, 2), 0.5)
    assert gmm_overlap_mass(flat)["frac_below"] == pytest.approx(1.0)
    assert gmm_overlap_mass(flat)["passes"] is False
    assert GMM_OVERLAP_MAX == 0.25
    assert SPATIAL_CORR_MIN == 0.5
    with pytest.raises(ValueError):
        gmm_overlap_mass(np.array([[0.9, 0.5]]))
    with pytest.raises(ValueError):
        gmm_overlap_mass(np.zeros((0, 2)))
    assert GMM_OVERLAP_POSTERIOR == 0.8


def test_anchor_bar_just_above_best_control():
    out = anchor_min_delta(0.86, {"month": 0.40, "ccaa": 0.70})
    assert out["best_control"] == "ccaa"
    assert out["recommended_bar"] == pytest.approx(0.75)
    assert out["candidate_beats_bar"] is True
    out2 = anchor_min_delta(0.71, {"month": 0.40, "ccaa": 0.70})
    assert out2["candidate_beats_bar"] is False


def test_validation_verdict_logic():
    assert validation_verdict({"a": True}, {"b": True}) == "accepted"
    assert validation_verdict({"a": False}, {"b": True}) == "rejected"
    assert validation_verdict({"a": True}, {"b": False}) == "inconclusive"
    with pytest.raises(ValueError):
        validation_verdict({}, {"b": True})
