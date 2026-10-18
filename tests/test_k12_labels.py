"""Contrato de la tipología k12 elegida (03e -> Plan B §B.2.1/§B.2.2).

La política de reajuste se extrae de ``notebooks/03e_validacion.ipynb``
(celdas 3/17): StandardScaler(PC1–7), KMeans k=12 para la etiqueta dura,
GMM de covarianza completa k=12 para posteriores, semilla 42. Los objetos
ajustados sobre positivos puntúan el fondo sin reajustar nunca sobre él.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.typology import (
    K12_ELECTED_K,
    K12_PC_COLUMNS,
    fit_k12_typology,
)


def _features(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """PC1–7 con 12 blobs bien separados (una tipología recuperable)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(K12_ELECTED_K, len(K12_PC_COLUMNS))) * 8
    assign = np.tile(np.arange(K12_ELECTED_K), int(np.ceil(n / K12_ELECTED_K)))[:n]
    xs = centers[assign] + rng.normal(scale=0.1, size=(n, len(K12_PC_COLUMNS)))
    return pd.DataFrame(
        {
            "detection_id": [f"det_{i:05d}" for i in range(n)],
            **dict(zip(K12_PC_COLUMNS, xs.T)),
        }
    )


class TestFitK12Typology:
    def test_labels_cover_all_k_groups_on_separable_data(self):
        typology = fit_k12_typology(_features())
        labels = typology.labels_for(_features())

        assert list(labels.columns[:2]) == ["detection_id", "cluster_meteo"]
        assert labels["cluster_meteo"].between(0, K12_ELECTED_K - 1).all()
        assert labels["cluster_meteo"].nunique() == K12_ELECTED_K

    def test_posteriors_are_probabilities(self):
        labels = fit_k12_typology(_features()).labels_for(_features())
        posterior_cols = [c for c in labels.columns if c.startswith("gmm_k12_full_p")]

        assert len(posterior_cols) == K12_ELECTED_K
        assert np.allclose(labels[posterior_cols].to_numpy().sum(axis=1), 1.0)
        assert (labels[posterior_cols].to_numpy() >= 0).all()

    def test_deterministic_under_seed_42(self):
        a = fit_k12_typology(_features()).labels_for(_features())
        b = fit_k12_typology(_features()).labels_for(_features())

        pd.testing.assert_frame_equal(a, b)

    def test_keyed_by_detection_id_not_row_order(self):
        features = _features()
        shuffled = features.sample(frac=1.0, random_state=7).reset_index(drop=True)
        typology = fit_k12_typology(features)

        labels_shuffled = typology.labels_for(shuffled).set_index("detection_id")
        labels_plain = typology.labels_for(features).set_index("detection_id")

        pd.testing.assert_frame_equal(
            labels_shuffled.sort_index(), labels_plain.sort_index()
        )

    def test_missing_detection_id_raises(self):
        with pytest.raises(KeyError, match="detection_id"):
            fit_k12_typology(_features().drop(columns=["detection_id"]))

    def test_missing_pc_column_raises(self):
        with pytest.raises(KeyError, match="PC"):
            fit_k12_typology(_features().drop(columns=["PC7"]))

    def test_nan_pc_values_raise(self):
        features = _features()
        features.loc[0, "PC1"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fit_k12_typology(features)


class TestScoreBackground:
    """El fondo se puntúa con los objetos ajustados en positivos, sin reajustar."""

    def test_background_labels_use_fitted_objects(self):
        positives = _features(seed=0)
        background = _features(seed=1)  # different rows, same PC schema
        typology = fit_k12_typology(positives)

        scored = typology.score_background(background)

        assert scored["detection_id"].tolist() == background["detection_id"].tolist()
        assert scored["cluster_meteo"].between(0, K12_ELECTED_K - 1).all()
        posterior_cols = [c for c in scored.columns if c.startswith("gmm_k12_full_p")]
        assert np.allclose(scored[posterior_cols].to_numpy().sum(axis=1), 1.0)

    def test_score_background_is_labels_for(self):
        assert fit_k12_typology(_features()).score_background is not None


class TestAgainstRealData:
    """Sustituye al refit inline de 03e: mismas etiquetas duras k12."""

    def test_reproduces_03e_kmeans_k12_labels(self):
        from wildfire.data.io import load_csv

        try:
            features = load_csv("data/processed/unsupervised/features_a.csv")
        except FileNotFoundError:
            pytest.skip("03 artifacts not on disk")

        # 03e cell 3 refit: KMeans k12 on StandardScaler(PC1–7), seed 42.
        Xa = features[K12_PC_COLUMNS].to_numpy(dtype=float)
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        from sklearn.preprocessing import StandardScaler

        Xs = StandardScaler().fit_transform(Xa)
        expected = KMeans(
            n_clusters=K12_ELECTED_K, random_state=42, n_init=10
        ).fit_predict(Xs)

        labels = fit_k12_typology(features).labels_for(features)
        ari = adjusted_rand_score(expected, labels["cluster_meteo"].to_numpy())

        assert ari == pytest.approx(1.0), (
            f"k12 labels diverge from 03e refit (ARI={ari})"
        )
