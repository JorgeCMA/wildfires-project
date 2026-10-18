"""Contrato de reintegración 03 -> 04 (Plan B §B.2.2/§B.2.3/§B.2.4).

Toda la reintegración se ajusta sobre positivos y puntúa el fondo con los
mismos objetos (nunca reajusta). ``score_anomalia``: media de las tres
puntuaciones z-scoreadas, min-max a [0,1] (PLAN §B.2.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildfire.processing.occurrence import (
    CANDIDATE_FEATURES,
    CANDIDATE_POSTERIOR_FEATURES,
    assemble_modeling_frame,
)
from wildfire.processing.reintegration import (
    ANOMALY_SCORE_COLS,
    SCORE_ANOMALIA_COL,
    fit_background_scorer,
    score_anomalia,
)
from wildfire.processing.typology import (
    K12_ELECTED_K,
    K12_POSTERIOR_COLUMNS,
)

PC_COLUMNS = [f"PC{i}" for i in range(1, 8)]
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "vapour_pressure_deficit",
    "et0_fao_evapotranspiration",
    "shortwave_radiation",
    "boundary_layer_height",
    "surface_pressure",
    "wind_speed_10m",
    "wind_gusts_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "wind_direction_10m",
]


def _weather_rows(n: int, seed: int) -> pd.DataFrame:
    """Raw-weather rows living in a 7-effective-dimensional space.

    Latent rank 7 by construction: the weather PCA (target_variance from
    config) must select exactly PC1–7, the space the k12 typology lives in.
    """
    rng = np.random.default_rng(seed)
    # Orthonormal mixing: the column space stays isotropic, so the 7 latents
    # split PCA variance evenly (PC7 ratio ≈ 1/7 > the 0.10 slack needed for
    # the 0.90 target to keep all 7 PCs at any n).
    q, _ = np.linalg.qr(rng.normal(size=(len(WEATHER_COLUMNS), 7)))
    latent = rng.normal(size=(n, 7))
    xs = latent @ q.T + rng.normal(scale=0.05, size=(n, len(WEATHER_COLUMNS)))
    df = pd.DataFrame({c: xs[:, i] for i, c in enumerate(WEATHER_COLUMNS)})
    df["precipitation"] = np.abs(df["precipitation"])  # log1p input domain
    df["wind_direction_10m"] = np.mod(df["wind_direction_10m"], 360.0)
    return df


def _positives(n: int = 120, seed: int = 0) -> pd.DataFrame:
    df = _weather_rows(n, seed)
    df.insert(0, "detection_id", [f"det_{i:05d}" for i in range(n)])
    return df


class TestScoreAnomalia:
    def test_formula_minmax_and_direction(self):
        df = pd.DataFrame(
            {
                "isolation_forest_score": [0.0, 1.0, 2.0, 3.0],
                "lof_score": [0.0, 0.5, 2.5, 3.0],
                "mahalanobis_score": [3.0, 2.0, 1.0, 0.0],
            }
        )
        s = score_anomalia(df)

        assert s.name == SCORE_ANOMALIA_COL
        assert s.between(0.0, 1.0).all()
        # La fila más rara en las tres escalas (índice 3) puntúa más alto.
        assert s.idxmax() == 3

    def test_missing_column_raises(self):
        with pytest.raises(KeyError, match="lof_score"):
            score_anomalia(pd.DataFrame({"isolation_forest_score": [1.0]}))

    def test_constant_columns_give_zero(self):
        df = pd.DataFrame({c: [1.0] * 4 for c in ANOMALY_SCORE_COLS})
        assert (score_anomalia(df) == 0.0).all()


class TestBackgroundScorer:
    def test_score_adds_candidate_columns(self):
        scorer = fit_background_scorer(_positives())
        background = _weather_rows(24, seed=99).assign(
            background_id=[f"bg_{i:05d}" for i in range(24)]
        )

        scored = scorer.score(background)

        assert list(scored.columns) == [
            "background_id",
            "cluster_meteo",
            SCORE_ANOMALIA_COL,
        ]
        assert scored["cluster_meteo"].between(0, K12_ELECTED_K - 1).all()
        assert scored[SCORE_ANOMALIA_COL].between(0.0, 1.0).all()

    def test_score_include_posteriors_appends_twelve_columns(self):
        scorer = fit_background_scorer(_positives())
        background = _weather_rows(24, seed=99).assign(
            background_id=[f"bg_{i:05d}" for i in range(24)]
        )

        scored = scorer.score(background, include_posteriors=True)

        assert list(scored.columns) == [
            "background_id",
            "cluster_meteo",
            SCORE_ANOMALIA_COL,
            *K12_POSTERIOR_COLUMNS,
        ]
        assert np.allclose(scored[K12_POSTERIOR_COLUMNS].to_numpy().sum(axis=1), 1.0)

    def test_deterministic_under_seed_42(self):
        background = _weather_rows(12, seed=5).assign(
            background_id=[f"bg_{i:05d}" for i in range(12)]
        )
        a = fit_background_scorer(_positives()).score(background)
        b = fit_background_scorer(_positives()).score(background)
        pd.testing.assert_frame_equal(a, b)

    def test_background_never_in_fit_scoring_is_stable(self):
        """Añadir filas al fondo no cambia las puntuaciones de las demás."""
        scorer = fit_background_scorer(_positives())
        base = _weather_rows(12, seed=5).assign(
            background_id=[f"bg_{i:05d}" for i in range(12)]
        )
        extra = _weather_rows(6, seed=6).assign(
            background_id=[f"bgx_{i:05d}" for i in range(6)]
        )
        extended = pd.concat([base, extra], ignore_index=True)

        a = scorer.score(base)
        b = scorer.score(extended).iloc[: len(base)]

        pd.testing.assert_frame_equal(a, b)

    def test_scores_positives_by_detection_id(self):
        positives = _positives(48)
        scored = fit_background_scorer(positives).score(positives.head(10))

        assert (
            scored["detection_id"].tolist()
            == positives["detection_id"].head(10).tolist()
        )

    def test_missing_weather_column_raises(self):
        scorer = fit_background_scorer(_positives())
        background = _weather_rows(5, seed=1).assign(
            background_id=[f"bg_{i}" for i in range(5)]
        )
        with pytest.raises(KeyError, match="temperature_2m"):
            scorer.score(background.drop(columns=["temperature_2m"]))

    def test_missing_id_column_raises(self):
        scorer = fit_background_scorer(_positives())
        with pytest.raises(KeyError, match="id column"):
            scorer.score(_weather_rows(5, seed=1))


class TestCandidateFrameContract:
    def _frames(self, with_candidates: bool):
        from tests.test_occurrence import _make_frame

        positives = _make_frame(seed=3)
        background = _make_frame(seed=4)
        pos_c = neg_c = None
        if with_candidates:
            pos_c = pd.DataFrame(
                {
                    "score_anomalia": np.linspace(0, 1, len(positives)),
                    "cluster_meteo": 0,
                }
            )
            neg_c = pd.DataFrame(
                {
                    "score_anomalia": np.linspace(1, 0, len(background)),
                    "cluster_meteo": 1,
                }
            )
        return positives, background, pos_c, neg_c

    def test_without_candidates_frame_unchanged(self):
        from wildfire.processing.occurrence import EVERYDAY_FEATURES, LABEL_COL

        positives, background, _, _ = self._frames(with_candidates=False)
        frame = assemble_modeling_frame(positives, background)

        assert list(frame.columns) == [*EVERYDAY_FEATURES, LABEL_COL]

    def test_with_candidates_appends_in_canonical_order(self):
        from wildfire.processing.occurrence import EVERYDAY_FEATURES, LABEL_COL

        positives, background, pos_c, neg_c = self._frames(with_candidates=True)
        frame = assemble_modeling_frame(
            positives, background, pos_candidates=pos_c, neg_candidates=neg_c
        )

        assert list(frame.columns) == [
            *EVERYDAY_FEATURES,
            *CANDIDATE_FEATURES,
            LABEL_COL,
        ]
        n = len(positives)
        assert frame.loc[0, "cluster_meteo"] == 0
        assert frame.loc[n, "cluster_meteo"] == 1

    def test_candidate_column_order_is_canonical_not_input_order(self):
        positives, background, pos_c, neg_c = self._frames(with_candidates=True)
        # Invertir el orden en la entrada: la salida sigue CANDIDATE_FEATURES.
        frame = assemble_modeling_frame(
            positives,
            background,
            pos_candidates=pos_c[["cluster_meteo", "score_anomalia"]],
            neg_candidates=neg_c[["cluster_meteo", "score_anomalia"]],
        )
        n = len(positives)
        assert frame["score_anomalia"].iloc[0] == 0.0
        assert frame["score_anomalia"].iloc[n] == 1.0

    def test_posterior_candidates_allowed(self):
        from wildfire.processing.occurrence import EVERYDAY_FEATURES, LABEL_COL

        positives, background, _, _ = self._frames(with_candidates=False)
        n, m = len(positives), len(background)
        pos_c = pd.DataFrame(
            {
                "score_anomalia": np.linspace(0, 1, n),
                **{c: 1.0 / len(K12_POSTERIOR_COLUMNS) for c in K12_POSTERIOR_COLUMNS},
            }
        )
        neg_c = pd.DataFrame(
            {
                "score_anomalia": np.linspace(1, 0, m),
                **{c: 1.0 / len(K12_POSTERIOR_COLUMNS) for c in K12_POSTERIOR_COLUMNS},
            }
        )
        frame = assemble_modeling_frame(
            positives, background, pos_candidates=pos_c, neg_candidates=neg_c
        )

        expected = [
            c
            for c in [*CANDIDATE_FEATURES, *CANDIDATE_POSTERIOR_FEATURES]
            if c in pos_c.columns
        ]
        assert list(frame.columns) == [*EVERYDAY_FEATURES, *expected, LABEL_COL]

    def test_only_one_candidate_frame_raises(self):
        positives, background, pos_c, _ = self._frames(with_candidates=True)
        with pytest.raises(ValueError, match="come together"):
            assemble_modeling_frame(positives, background, pos_candidates=pos_c)

    def test_unknown_candidate_column_raises(self):
        positives, background, pos_c, neg_c = self._frames(with_candidates=True)
        bad = pos_c.assign(satellite_trick=1.0)
        with pytest.raises(ValueError, match="candidate set"):
            assemble_modeling_frame(
                positives, background, pos_candidates=bad, neg_candidates=neg_c
            )

    def test_length_mismatch_raises(self):
        positives, background, pos_c, neg_c = self._frames(with_candidates=True)
        with pytest.raises(ValueError, match="rows"):
            assemble_modeling_frame(
                positives,
                background,
                pos_candidates=pos_c.iloc[:-1],
                neg_candidates=neg_c,
            )

    def test_satellite_column_in_candidates_raises(self):
        positives, background, pos_c, neg_c = self._frames(with_candidates=True)
        bad_p = pos_c.assign(frp=1.0)
        bad_n = neg_c.assign(frp=1.0)
        with pytest.raises(ValueError, match="satellite"):
            assemble_modeling_frame(
                positives, background, pos_candidates=bad_p, neg_candidates=bad_n
            )
