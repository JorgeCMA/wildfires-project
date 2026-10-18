"""Reintegration of unsupervised artifacts (block 03) into the supervised
occurrence model (Plan B of planning/plan_supervised_implementation.md).

Contract (PLAN §B.2.2/§B.2.3, §3.2 of plan_unsupervised_next_implementations.md):

- Everything is **fit on positives only**; background rows are *scored* by the
  fitted objects, never included in any fit.
- ``cluster_meteo`` comes from the elected k12 typology (Step 1:
  :class:`~wildfire.processing.typology.K12Typology`).
- ``score_anomalia`` (PLAN §B.2.3 default): row-wise mean of the three z-scored
  detector scores (isolation forest, LOF, Mahalanobis), min-max rescaled to
  [0, 1]. For positives the documented source is ``anomalias_{a,b}.csv``
  (:func:`load_unsupervised_scores` + :func:`score_anomalia`); for background
  the detectors are refit on positives and applied to background rows
  (:class:`BackgroundScorer`).

Set A (PC1–7, ``pca.target_variance`` from config) is the shared space: the
weather-PCA pipeline projects raw weather rows into it, and both the k12
typology and the anomaly detectors live on those PC scores — the same space
03b/03c used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from wildfire.processing.pca_weather import fit_weather_pca
from wildfire.processing.typology import (
    K12_PC_COLUMNS,
    K12_POSTERIOR_COLUMNS,
    K12Typology,
    fit_k12_typology,
)
from wildfire.processing.unsupervised import CLUSTERING_SEED, ID_COL

if TYPE_CHECKING:
    from sklearn.covariance import MinCovDet  # type: ignore[import-untyped]
    from sklearn.ensemble import IsolationForest  # type: ignore[import-untyped]
    from sklearn.neighbors import LocalOutlierFactor  # type: ignore[import-untyped]
    from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

#: Detector score columns of ``anomalias_{a,b}.csv`` (03c output).
ANOMALY_SCORE_COLS: list[str] = [
    "isolation_forest_score",
    "lof_score",
    "mahalanobis_score",
]

#: Continuous reintegration candidate written by this module (PLAN §B.2.4).
SCORE_ANOMALIA_COL = "score_anomalia"

#: Background rows are keyed by ``background_id`` (PLAN §A file names).
BACKGROUND_ID_COL = "background_id"

ANOMALIES_CSV = "data/processed/unsupervised/anomalias_{}.csv"


def _id_column(df: pd.DataFrame) -> str:
    if ID_COL in df.columns:
        return ID_COL
    if BACKGROUND_ID_COL in df.columns:
        return BACKGROUND_ID_COL
    raise KeyError(f"Missing id column: expected {ID_COL} or {BACKGROUND_ID_COL}")


def load_unsupervised_scores(anomaly_set: str = "a") -> pd.DataFrame:
    """Persisted anomaly scores of the positives (03c), keyed by detection_id.

    Never joins by row order: the file's only key is ``detection_id``.
    """
    from wildfire.data.io import load_csv

    df = load_csv(ANOMALIES_CSV.format(anomaly_set))
    if ID_COL not in df.columns:
        raise KeyError(f"{ANOMALIES_CSV.format(anomaly_set)} lacks {ID_COL}")
    return df


def _minmax(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not np.isfinite(hi - lo) or hi == lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def score_anomalia(df: pd.DataFrame) -> pd.Series:
    """PLAN §B.2.3 default consensus score, descriptive on the frame passed.

    z-scores each detector column (params fit on ``df``), averages the three
    row-wise, then min-max rescales to [0, 1] (range from ``df``). Higher =
    more anomalous. On ``load_unsupervised_scores()`` this reproduces the
    positives' ``score_anomalia``.
    """
    from sklearn.preprocessing import StandardScaler

    missing = [c for c in ANOMALY_SCORE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing anomaly score columns: {missing}")
    raw = df[ANOMALY_SCORE_COLS].to_numpy(dtype=float)
    z = StandardScaler().fit_transform(raw)
    mean = z.mean(axis=1)
    return pd.Series(
        np.clip(_minmax(mean, float(mean.min()), float(mean.max())), 0.0, 1.0),
        index=df.index,
        name=SCORE_ANOMALIA_COL,
    )


@dataclass
class BackgroundScorer:
    """Fitted-on-positives objects that score background rows (PLAN §B.2.2).

    Holds the weather-PCA pipeline (raw weather -> PC1..PCk, set A), the k12
    typology (:class:`K12Typology`), the three anomaly detectors (isolation
    forest, LOF in novelty mode, robust Mahalanobis/MinCovDet), and the
    normalization of ``score_anomalia`` (z-score + min-max params fit on the
    positives' detector scores). ``score`` never refits anything.
    """

    pca_pipeline: Pipeline
    n_components: int
    typology: K12Typology
    iforest: IsolationForest
    lof: LocalOutlierFactor  # novelty=True
    mcd: MinCovDet
    score_mean_scaler: StandardScaler  # fit on the 3 positives detector scores
    score_lo: float
    score_hi: float

    def _pc_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Raw weather rows -> PC scores via the pipeline fit on positives."""
        from wildfire.processing.pca_weather import pca_input_columns

        missing = set(pca_input_columns()) - set(df.columns)
        if missing:
            raise KeyError(f"Missing weather columns for PC scoring: {sorted(missing)}")
        frame = df[pca_input_columns()]
        return self.pca_pipeline.transform(frame)[:, : self.n_components]

    def score(
        self, df: pd.DataFrame, *, include_posteriors: bool = False
    ) -> pd.DataFrame:
        """``cluster_meteo`` + ``score_anomalia`` for the rows of ``df``.

        Works for positives (id = ``detection_id``) and background (id =
        ``background_id``): the id column present in ``df`` is carried through.
        With ``include_posteriors=True`` the 12 GMM posterior columns
        (``K12_POSTERIOR_COLUMNS``) are appended, the alternative encoding of
        the ``cluster_meteo`` candidate (PLAN §B.2.2).
        """
        id_col = _id_column(df)
        pcs = self._pc_scores(df)
        features = pd.DataFrame(pcs, columns=K12_PC_COLUMNS[: pcs.shape[1]])
        features[ID_COL] = df[id_col].to_numpy()

        labels = self.typology.labels_for(features).drop(columns=[ID_COL])

        detector_scores = np.column_stack(
            [
                -self.iforest.decision_function(pcs),
                -self.lof.decision_function(pcs),
                self.mcd.mahalanobis(pcs),
            ]
        )
        z = self.score_mean_scaler.transform(detector_scores)
        mean = z.mean(axis=1)
        out = pd.DataFrame({id_col: df[id_col].to_numpy()})
        out["cluster_meteo"] = labels["cluster_meteo"].to_numpy()
        out[SCORE_ANOMALIA_COL] = np.clip(
            _minmax(mean, self.score_lo, self.score_hi), 0.0, 1.0
        )
        if include_posteriors:
            out[K12_POSTERIOR_COLUMNS] = labels[K12_POSTERIOR_COLUMNS].to_numpy()
        return out


def fit_background_scorer(
    positives: pd.DataFrame,
    seed: int = CLUSTERING_SEED,
) -> BackgroundScorer:
    """Fit every reintegration object on the positives frame.

    Parameters
    ----------
    positives:
        Fire rows with ``detection_id`` and the raw weather columns
        (:func:`pca_input_columns`). The weather-PCA pipeline, the k12
        typology, the anomaly detectors and the ``score_anomalia``
        normalization are all fit here and only here.

    Returns
    -------
    BackgroundScorer
        Ready to ``score`` background (or positives) rows.
    """
    from sklearn.covariance import MinCovDet
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler

    if ID_COL not in positives.columns:
        raise KeyError(f"Missing column: {ID_COL}")

    pca = fit_weather_pca(positives)
    if pca.n_components < len(K12_PC_COLUMNS):
        raise ValueError(
            f"k12 typology lives on PC1–{len(K12_PC_COLUMNS)} but the PCA "
            f"selected only {pca.n_components} (target_variance): the config "
            "or the typology drifted, they must share one space"
        )
    pcs = pca.scores[K12_PC_COLUMNS].to_numpy(dtype=float)
    features = pd.DataFrame(pcs, columns=K12_PC_COLUMNS)
    features[ID_COL] = positives[ID_COL].to_numpy()

    typology = fit_k12_typology(features, seed=seed)

    iforest = IsolationForest(random_state=seed).fit(pcs)
    # Novelty mode: LOF that can score NEW rows (03c's fit_lof was in-sample
    # descriptive only; background is out-of-sample by construction).
    lof = LocalOutlierFactor(novelty=True).fit(pcs)
    mcd = MinCovDet(random_state=seed).fit(pcs)

    detector_scores = np.column_stack(
        [
            -iforest.decision_function(pcs),
            -lof.decision_function(pcs),
            mcd.mahalanobis(pcs),
        ]
    )
    score_mean_scaler = StandardScaler().fit(detector_scores)
    mean = score_mean_scaler.transform(detector_scores).mean(axis=1)
    return BackgroundScorer(
        pca_pipeline=pca.pipeline,
        n_components=pca.n_components,
        typology=typology,
        iforest=iforest,
        lof=lof,
        mcd=mcd,
        score_mean_scaler=score_mean_scaler,
        score_lo=float(mean.min()),
        score_hi=float(mean.max()),
    )
