"""Model units for the supervised occurrence model (Plan C.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.analysis.occurrence_model import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    categorical_columns,
    categorical_to_dummies,
    continuous_columns,
    fold_scaler_apply,
    fold_scaler_fit,
    model_candidates,
)
from wildfire.processing.occurrence import EVERYDAY_FEATURES


def _policy_frame(n=8, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in CONTINUOUS_FEATURES}
    data["ccaa"] = rng.choice(["Asturias", "Galicia"], n)
    data["clc_class"] = rng.choice([1, 2, 3], n)
    return pd.DataFrame(data)[EVERYDAY_FEATURES]


class TestColumnPolicy:
    def test_partition_of_everyday(self):
        assert set(CATEGORICAL_FEATURES).isdisjoint(CONTINUOUS_FEATURES)
        assert set(CATEGORICAL_FEATURES) | set(CONTINUOUS_FEATURES) == set(
            EVERYDAY_FEATURES
        )

    def test_categorical_default(self):
        assert categorical_columns() == ["ccaa", "clc_class"]

    def test_candidates_added(self):
        assert "cluster_meteo" in categorical_columns(include_candidates=True)

    def test_continuous_columns_includes_candidate(self):
        frame = _policy_frame()
        frame["score_anomalia"] = 0.5
        assert "score_anomalia" in continuous_columns(frame)


class TestModelCandidates:
    def test_candidate_set(self):
        models = model_candidates()
        assert set(models) == {
            "logistic",
            "logistic_balanced",
            "random_forest",
            "random_forest_balanced",
            "hist_gradient_boosting",
        }

    def test_balanced_variants_set_class_weight(self):
        models = model_candidates()
        assert models["logistic_balanced"].class_weight == "balanced"
        assert models["random_forest_balanced"].class_weight == "balanced_subsample"
        assert models["logistic"].class_weight is None

    def test_fresh_instance_each_call(self):
        first = model_candidates(seed=1)
        second = model_candidates(seed=1)
        assert first["logistic"] is not second["logistic"]
        assert first["random_forest"] is not second["random_forest"]

    def test_seed_propagates(self):
        models = model_candidates(seed=7)
        assert models["logistic"].random_state == 7
        assert models["random_forest"].random_state == 7
        assert models["hist_gradient_boosting"].random_state == 7


class TestCategoricalToDummies:
    def test_train_columns(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "ccaa": ["A", "B"]})
        encoded, columns = categorical_to_dummies(frame, ["ccaa"])
        assert columns == ["ccaa_A", "ccaa_B"]
        assert "ccaa" not in encoded.columns
        assert encoded[["ccaa_A", "ccaa_B"]].to_numpy().tolist() == [[1, 0], [0, 1]]

    def test_apply_unseen_is_all_zero(self):
        train = pd.DataFrame({"ccaa": ["A", "B"], "x": [1.0, 2.0]})
        _, columns = categorical_to_dummies(train, ["ccaa"])
        test = pd.DataFrame({"ccaa": ["C"], "x": [3.0]})
        encoded, same = categorical_to_dummies(test, ["ccaa"], train_columns=columns)
        assert same == columns
        assert encoded[columns].to_numpy().tolist() == [[0, 0]]

    def test_missing_column_skipped(self):
        frame = pd.DataFrame({"x": [1.0], "ccaa": ["A"]})
        encoded, columns = categorical_to_dummies(frame, ["ccaa", "cluster_meteo"])
        assert columns == ["ccaa_A"]
        assert "cluster_meteo" not in encoded.columns

    def test_numeric_categories(self):
        frame = pd.DataFrame({"clc_class": [2, 1]})
        _, columns = categorical_to_dummies(frame, ["clc_class"])
        assert columns == ["clc_class_1", "clc_class_2"]

    def test_non_categorical_untouched(self):
        frame = pd.DataFrame({"x": [1.0, 2.0], "ccaa": ["A", "A"]})
        encoded, _ = categorical_to_dummies(frame, ["ccaa"])
        assert encoded["x"].tolist() == [1.0, 2.0]


class TestFoldScaler:
    def test_scales_only_given_columns(self):
        frame = _policy_frame(20)
        frame["ccaa"] = "A"
        cols = continuous_columns(frame)
        scaler = fold_scaler_fit(frame, cols)
        scaled = fold_scaler_apply(frame, scaler, cols)
        assert np.allclose(scaled[cols].to_numpy().mean(axis=0), 0.0, atol=1e-9)
        assert scaled["ccaa"].tolist() == frame["ccaa"].tolist()

    def test_missing_column_raises(self):
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        with pytest.raises(KeyError):
            fold_scaler_fit(frame, ["x", "missing"])

    def test_no_columns_raises(self):
        with pytest.raises(ValueError):
            fold_scaler_fit(pd.DataFrame({"ccaa": ["A"]}), [])
