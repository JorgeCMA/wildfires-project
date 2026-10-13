import numpy as np
import pandas as pd
import pytest

from wildfire.processing.occurrence import (
    EVERYDAY_FEATURES,
    LABEL_COL,
    SATELLITE_BLOCKLIST,
    add_season,
    assemble_modeling_frame,
    background_skeleton,
    blocked_splits,
    occurrence_features,
    occurrence_settings,
    rank_metrics,
)


def _make_frame(n=40, seed=3, extra=None):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0, 1, n) for c in EVERYDAY_FEATURES}
    data["ccaa"] = rng.choice(["Asturias", "Galicia"], n)
    data["clc_uniform_surroundings"] = rng.choice([True, False], n)
    data["wind_direction_10m"] = rng.uniform(0, 360, n)
    frame = pd.DataFrame(data).drop(columns=["wd_sin", "wd_cos"])
    if extra:
        for key, value in extra.items():
            frame[key] = value
    return frame


class TestSettings:
    def test_defaults(self):
        settings = occurrence_settings()
        assert settings["background_ratio"] == 2
        assert settings["random_state"] == 42
        assert settings["cv_groups"] == ["season", "ccaa"]

    def test_invalid_ratio_rejected(self, monkeypatch):
        import wildfire.processing.occurrence as occ

        monkeypatch.setattr(
            occ,
            "load_config",
            lambda: {
                "modeling": {
                    "occurrence": {
                        "background_ratio": 0,
                        "random_state": 42,
                        "cv_groups": ["ccaa"],
                    }
                }
            },
        )
        with pytest.raises(ValueError):
            occ.occurrence_settings()


class TestOccurrenceFeatures:
    def test_fixed_order_and_count(self):
        out = occurrence_features(_make_frame())
        assert list(out.columns) == EVERYDAY_FEATURES

    def test_wind_direction_derived(self):
        out = occurrence_features(_make_frame())
        assert np.isfinite(out[["wd_sin", "wd_cos"]].to_numpy()).all()
        assert "wind_direction_10m" not in out.columns

    def test_missing_wind_raises(self):
        df = _make_frame().drop(columns=["wind_direction_10m"])
        with pytest.raises(KeyError):
            occurrence_features(df)

    def test_missing_feature_raises(self):
        df = _make_frame().drop(columns=["temperature_2m"])
        with pytest.raises(KeyError):
            occurrence_features(df)

    def test_satellite_column_warns_and_drops(self):
        df = _make_frame(extra={"frp": 5.0, "confidence_num": 80.0})
        with pytest.warns(UserWarning):
            out = occurrence_features(df)
        assert "frp" not in out.columns
        assert "confidence_num" not in out.columns


class TestAddSeason:
    def test_quarter_format(self):
        df = pd.DataFrame({"acq_date": ["2023-01-15", "2024-11-02"]})
        out = add_season(df)
        assert out["season"].tolist() == ["2023-Q1", "2024-Q4"]

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            add_season(pd.DataFrame({"a": [1]}))

    def test_bad_date_raises(self):
        with pytest.raises(ValueError):
            add_season(pd.DataFrame({"acq_date": ["not-a-date"]}))


class TestAssemble:
    def test_labels_and_shape(self):
        frame = assemble_modeling_frame(_make_frame(10), _make_frame(20, seed=9))
        assert frame[LABEL_COL].tolist() == [1] * 10 + [0] * 20
        assert len(frame) == 30

    def test_satellite_in_positives_raises(self):
        pos = _make_frame(10, extra={"brightness": 300.0})
        with pytest.raises(ValueError):
            assemble_modeling_frame(pos, _make_frame(10, seed=9))

    def test_satellite_in_background_raises(self):
        with pytest.raises(ValueError):
            assemble_modeling_frame(
                _make_frame(10), _make_frame(10, seed=9, extra={"frp": 1.0})
            )

    def test_background_missing_feature_raises(self):
        with pytest.raises(KeyError):
            assemble_modeling_frame(
                _make_frame(10), _make_frame(10, seed=9).drop(columns=["ccaa"])
            )


class TestSkeleton:
    def test_shape_and_determinism(self):
        first = background_skeleton(["A", "B"], "2023-01-01", "2023-01-10", seed=1)
        second = background_skeleton(["A", "B"], "2023-01-01", "2023-01-10", seed=1)
        assert len(first) == 10
        pd.testing.assert_frame_equal(first, second)

    def test_n_per_day(self):
        grid = background_skeleton(["A"], "2023-01-01", "2023-01-05", n_per_day=3)
        assert len(grid) == 15

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            background_skeleton([], "2023-01-01", "2023-01-10")
        with pytest.raises(ValueError):
            background_skeleton(["A"], "2023-01-10", "2023-01-01")


class TestBlockedSplits:
    def test_covers_all_groups_disjoint(self):
        df = _make_frame(40)
        df["season"] = ["2023-Q1"] * 20 + ["2023-Q2"] * 20
        splits = blocked_splits(df, "season")
        assert len(splits) == 2
        for split in splits:
            assert len(set(split.train_index) & set(split.test_index)) == 0
            assert len(split.test_index) == 20

    def test_missing_group_raises(self):
        with pytest.raises(KeyError):
            blocked_splits(_make_frame(10), "season")


class TestRankMetrics:
    def test_perfect_ranking(self):
        metrics = rank_metrics(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert metrics["pr_auc"] == pytest.approx(1.0)
        assert metrics["roc_auc"] == pytest.approx(1.0)

    def test_single_label_raises(self):
        with pytest.raises(ValueError):
            rank_metrics(pd.Series([1, 1]), np.array([0.5, 0.6]))

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            rank_metrics(pd.Series([0, 1]), np.array([0.5, np.nan]))

    def test_satellite_blocklist_covers_confidence_family(self):
        for col in ("confidence_num", "confidence_cat", "frp", "brightness"):
            assert col in SATELLITE_BLOCKLIST
