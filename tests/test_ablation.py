"""Ablation verdict unit (PLAN §C.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.analysis.ablation import (
    ABLATION_SEED,
    ablation_verdict,
    bootstrap_mean_ci,
)


class TestBootstrapMeanCi:
    def test_reproducible(self):
        values = [0.1, 0.2, 0.3, 0.4]
        first = bootstrap_mean_ci(values, seed=1)
        second = bootstrap_mean_ci(values, seed=1)
        assert first == second

    def test_brackets_mean(self):
        values = np.linspace(-0.5, 0.5, 50)
        low, high = bootstrap_mean_ci(values, n_boot=500, seed=ABLATION_SEED)
        assert low <= values.mean() <= high

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_mean_ci([])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            bootstrap_mean_ci([0.1, np.nan])


class TestAblationVerdict:
    def test_positive_delta_promoted(self):
        with_metrics = [0.5, 0.6, 0.55, 0.62, 0.58]
        without_metrics = [0.4, 0.45, 0.42, 0.5, 0.44]
        verdict = ablation_verdict(with_metrics, without_metrics)
        assert verdict["promoted"] is True
        assert verdict["mean_delta"] > 0
        assert verdict["ci_low"] > 0

    def test_negative_delta_not_promoted(self):
        verdict = ablation_verdict([0.3, 0.32], [0.4, 0.42])
        assert verdict["promoted"] is False

    def test_margin_gate(self):
        with_metrics = [0.41, 0.42, 0.43, 0.44, 0.45]
        without_metrics = [0.4, 0.41, 0.42, 0.43, 0.44]
        assert ablation_verdict(with_metrics, without_metrics)["promoted"] is True
        assert (
            ablation_verdict(with_metrics, without_metrics, margin=0.05)["promoted"]
            is False
        )

    def test_negative_protocol_blocks(self):
        verdict = ablation_verdict(
            {"season": [0.6, 0.61], "ccaa": [0.3, 0.31]},
            {"season": [0.4, 0.41], "ccaa": [0.4, 0.41]},
        )
        assert verdict["per_protocol_mean"]["ccaa"] < 0
        assert verdict["promoted"] is False

    def test_per_protocol_min_reported(self):
        verdict = ablation_verdict([0.5, 0.4, 0.6], [0.4, 0.45, 0.5])
        assert verdict["per_protocol_min"]["default"] == pytest.approx(-0.05)
        assert verdict["per_fold_min"] == pytest.approx(-0.05)
        assert verdict["n_folds"] == 3

    def test_frame_matches_mapping(self):
        frame = pd.DataFrame(
            {
                "protocol": ["season", "season", "ccaa", "ccaa"],
                "pr_auc": [0.5, 0.6, 0.55, 0.65],
            }
        )
        without = pd.DataFrame(
            {
                "protocol": ["season", "season", "ccaa", "ccaa"],
                "pr_auc": [0.4, 0.5, 0.45, 0.55],
            }
        )
        from_frame = ablation_verdict(frame, without)
        from_map = ablation_verdict(
            {"season": [0.5, 0.6], "ccaa": [0.55, 0.65]},
            {"season": [0.4, 0.5], "ccaa": [0.45, 0.55]},
        )
        assert from_frame["promoted"] == from_map["promoted"]
        assert from_frame["mean_delta"] == pytest.approx(from_map["mean_delta"])

    def test_protocol_mismatch_raises(self):
        with pytest.raises(ValueError):
            ablation_verdict({"season": [0.5]}, {"ccaa": [0.4]})

    def test_fold_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            ablation_verdict({"season": [0.5, 0.6]}, {"season": [0.4]})

    def test_frame_without_pr_auc_raises(self):
        with pytest.raises(KeyError):
            ablation_verdict(pd.DataFrame({"x": [1.0]}), [0.5])
