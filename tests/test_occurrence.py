import numpy as np
import pandas as pd
import pytest

from wildfire.processing.occurrence import (
    BACKGROUND_ID_COL,
    EVERYDAY_FEATURES,
    LABEL_COL,
    SATELLITE_BLOCKLIST,
    add_season,
    assemble_modeling_frame,
    background_skeleton,
    blocked_splits,
    drop_leaking_background,
    load_background_days,
    occurrence_features,
    occurrence_settings,
    rank_metrics,
    sample_background,
    validate_background_days,
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


def _make_background(n=30, seed=5, acq_date="2023-06-01", **extra):
    df = _make_frame(n, seed=seed)
    df[BACKGROUND_ID_COL] = [f"bg{i:04d}" for i in range(n)]
    df["acq_date"] = acq_date
    for key, value in extra.items():
        df[key] = value
    return df


def _with_coords(df, lat, lon):
    out = df.copy()
    out["latitude"] = lat
    out["longitude"] = lon
    return out


class TestSettingsLeakage:
    def test_leakage_default(self):
        assert occurrence_settings()["leakage_max_km"] == 1.0

    def test_invalid_leakage_rejected(self, monkeypatch):
        import wildfire.processing.occurrence as occ

        monkeypatch.setattr(
            occ,
            "load_config",
            lambda: {
                "modeling": {
                    "occurrence": {
                        "background_ratio": 2,
                        "random_state": 42,
                        "cv_groups": ["ccaa"],
                        "leakage_max_km": -1.0,
                    }
                }
            },
        )
        with pytest.raises(ValueError):
            occ.occurrence_settings()


class TestLoadBackgroundDays:
    def test_missing_key_column_raises(self, tmp_path, monkeypatch):
        import wildfire.processing.occurrence as occ

        csv = tmp_path / "bg.csv"
        _make_background(5).drop(columns=[BACKGROUND_ID_COL]).to_csv(csv, index=False)
        monkeypatch.setattr(
            occ, "load_config", lambda: {"output": {"background_days": str(csv)}}
        )
        with pytest.raises(KeyError):
            load_background_days()

    def test_explicit_path_loads(self, tmp_path):
        csv = tmp_path / "bg.csv"
        _make_background(5).to_csv(csv, index=False)
        out = load_background_days(csv)
        assert BACKGROUND_ID_COL in out.columns
        assert len(out) == 5

    def test_missing_config_key_raises(self, monkeypatch):
        import wildfire.processing.occurrence as occ

        monkeypatch.setattr(occ, "load_config", lambda: {"output": {}})
        with pytest.raises(KeyError):
            load_background_days()


class TestValidateBackgroundDays:
    def _valid(self):
        background = _with_coords(_make_background(20), 40.0, -4.0)
        positives = _with_coords(_make_frame(5, seed=1), 0.0, 0.0)
        positives["acq_date"] = "2023-06-01"
        return background, positives

    def test_clean_passes_and_reports(self):
        background, positives = self._valid()
        report = validate_background_days(background, positives)
        assert report["n_rows"] == 20
        assert report["n_duplicate_ids"] == 0
        assert report["n_leakage"] == 0
        assert "nan_report" in report

    def test_duplicate_key_raises(self):
        background, positives = self._valid()
        background.loc[1, BACKGROUND_ID_COL] = background.loc[0, BACKGROUND_ID_COL]
        with pytest.raises(ValueError):
            validate_background_days(background, positives)

    def test_satellite_column_raises(self):
        background, positives = self._valid()
        background["brightness"] = 300.0
        with pytest.raises(ValueError):
            validate_background_days(background, positives)

    def test_weather_duplicates_allowed(self):
        background, positives = self._valid()
        background["rain"] = 0.0
        background["is_day"] = 1
        report = validate_background_days(background, positives)
        assert report["n_rows"] == 20

    def test_type_nonzero_raises(self):
        background, positives = self._valid()
        background["type"] = 1
        with pytest.raises(ValueError):
            validate_background_days(background, positives)

    def test_type_zero_allowed(self):
        background, positives = self._valid()
        background["type"] = 0
        validate_background_days(background, positives)

    def test_missing_feature_raises(self):
        background, positives = self._valid()
        with pytest.raises(KeyError):
            validate_background_days(
                background.drop(columns=["temperature_2m"]), positives
            )

    def test_missing_date_raises(self):
        background, positives = self._valid()
        with pytest.raises(KeyError):
            validate_background_days(background.drop(columns=["acq_date"]), positives)

    def test_same_day_collision_raises(self):
        background, positives = self._valid()
        positives = _with_coords(positives, 40.0, -4.0)
        background = _with_coords(background, 40.0, -4.0)
        with pytest.raises(ValueError):
            validate_background_days(background, positives)

    def test_same_day_collision_override(self):
        background, positives = self._valid()
        positives = _with_coords(positives, 40.0, -4.0)
        background = _with_coords(background, 40.0, -4.0)
        report = validate_background_days(background, positives, allow_leakage=True)
        assert report["n_leakage"] == len(background)

    def test_nan_coords_do_not_crash(self):
        background, positives = self._valid()
        background.loc[0, "latitude"] = np.nan
        report = validate_background_days(background, positives)
        assert report["nan_report"]["latitude"] == 1

    def test_different_day_not_leakage(self):
        background = _with_coords(
            _make_background(5, acq_date="2023-06-02"), 40.0, -4.0
        )
        positives = _with_coords(_make_frame(3, seed=2), 40.0, -4.0)
        positives["acq_date"] = "2023-06-01"
        report = validate_background_days(background, positives)
        assert report["n_leakage"] == 0


class TestDropLeakingBackground:
    def _frames(self):
        background = _with_coords(_make_background(10), 40.0, -4.0)
        positives = _with_coords(_make_frame(3, seed=1), 40.0, -4.0)
        positives["acq_date"] = "2023-06-01"
        return background, positives

    def test_drops_all_collisions(self):
        background, positives = self._frames()
        out = drop_leaking_background(background, positives)
        assert len(out) == 0

    def test_keeps_far_rows(self):
        background, positives = self._frames()
        background.loc[:4, "latitude"] = 50.0
        out = drop_leaking_background(background, positives)
        assert len(out) == 5

    def test_resets_index(self):
        background, positives = self._frames()
        background.loc[:4, "latitude"] = 50.0
        out = drop_leaking_background(background, positives)
        assert list(out.index) == list(range(len(out)))

    def test_noop_when_clean(self):
        background = _with_coords(
            _make_background(6, acq_date="2023-06-02"), 40.0, -4.0
        )
        positives = _with_coords(_make_frame(3, seed=2), 40.0, -4.0)
        positives["acq_date"] = "2023-06-01"
        out = drop_leaking_background(background, positives)
        assert len(out) == 6

    def test_output_passes_validation(self):
        background, positives = self._frames()
        background.loc[:4, "latitude"] = 50.0
        out = drop_leaking_background(background, positives)
        report = validate_background_days(out, positives)
        assert report["n_leakage"] == 0


class TestSampleBackground:
    def test_ratio_count(self):
        background = _make_background(100)
        out = sample_background(background, 10, ratio=2, seed=1)
        assert len(out) == 20

    def test_pool_smaller_returns_all(self):
        background = _make_background(5)
        out = sample_background(background, 10, ratio=2, seed=1)
        assert len(out) == 5

    def test_deterministic(self):
        background = _make_background(100)
        first = sample_background(background, 10, ratio=2, seed=7)
        second = sample_background(background, 10, ratio=2, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_sorted_by_id(self):
        background = _make_background(50)
        out = sample_background(background, 5, ratio=2, seed=3)
        ids = out[BACKGROUND_ID_COL].tolist()
        assert ids == sorted(ids)

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            sample_background(_make_background(5), 1, ratio=0, seed=1)

    def test_ratio_none_reads_config(self):
        out = sample_background(_make_background(100), 10, seed=1)
        assert len(out) == 20
