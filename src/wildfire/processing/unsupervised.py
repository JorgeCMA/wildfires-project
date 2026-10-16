"""Unsupervised-learning feature sets (block 03).

Single source of truth for the ENTRADA_*/PERFILADO column lists from
``planning/plan_unsupervised_implementation.md`` §5. Notebook 03a displays
and orchestrates; the units that can break silently live here, tested.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from wildfire.config import load_config
from wildfire.processing.occurrence import SATELLITE_BLOCKLIST
from wildfire.processing.pca_weather import AddWindCircular, pca_feature_columns

#: Fallback when ``unsupervised.min_cliff_delta`` is absent from config.
DEFAULT_MIN_CLIFF_DELTA = 0.2

#: Right-skewed columns needing ``log1p`` before scaling (plan §5).
LOG_COLS: list[str] = [
    "precipitation",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
]

#: Seasonality pairs, already in [-1, 1]: never scaled (plan §5, set B).
SEASON_SINCOS: list[str] = [
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

#: Sole representative of time of day (plan §5, set B).
TIME_COL = "franja_horaria"

#: The 8 CLC neighbour columns; directions collapse (plan §5, set C).
CLC_NEIGHBOURS: list[str] = [
    "clc_class_N",
    "clc_class_S",
    "clc_class_W",
    "clc_class_E",
    "clc_class_NW",
    "clc_class_NE",
    "clc_class_SW",
    "clc_class_SE",
]

#: Prefix for per-class neighbour-frequency columns built by
#: :func:`build_clc_environment`.
FREQ_VECINA_PREFIX = "freq_vecina_"

#: Fixed CLC-environment columns (frequencies are data-dependent, see
#: :func:`clc_environment_columns`).
CLC_ENV_FIXED: list[str] = [
    "pct_vecinos_igual_centro",
    "n_clases_distintas_vecinos",
]

#: Set A: the 14 weather-PCA features (12 base + wd pair).
ENTRADA_A: list[str] = pca_feature_columns()

#: Set B: A + seasonality + time of day.
ENTRADA_B: list[str] = [*ENTRADA_A, *SEASON_SINCOS, TIME_COL]

#: Set C (fixed part): B + centre pixel + collapsed neighbourhood.
#: Append :func:`clc_environment_columns` output for the frequency columns.
ENTRADA_C_FIXED: list[str] = [*ENTRADA_B, "clc_class", *CLC_ENV_FIXED]

#: Set D: pure spatial (hotspots, plan §5-D). Project to km in the notebook.
ENTRADA_D: list[str] = ["latitude", "longitude", "acq_date"]

#: Profiling / post-hoc validation only: never algorithm inputs (plan §5).
PERFILADO: list[str] = [
    "frp",
    "brightness",
    "brightness_ir",
    "scan",
    "confidence_num",
    "confidence_cat",
    "ccaa",
    "year",
    "month",
    "day_of_year",
    "daynight",
    "is_day",
    "acq_time",
    "acq_date",
    "type",
    "rain",
    "wind_direction_10m",
    "prevention_current",
    "prevention_prev3",
    "extinction_prev3",
    "ratio_prevencion_extincion",
    "euros_prevencion_por_ha",
    "euros_extincion_por_ha",
    "potencial_propagacion",
    "indice_riesgo_meteo",
    "indice_riesgo_meteo_log",
    "estres_termico_vegetacion",
    "indice_meteorologico",
    "riesgo_antropico_alto",
    "nivel_relativo",
    "componentes_disponibles",
]


def load_min_cliff_delta() -> float:
    """Read ``unsupervised.min_cliff_delta`` (plan §7B).

    Falls back to :data:`DEFAULT_MIN_CLIFF_DELTA` when the key is missing
    so older configs keep working.
    """
    try:
        value = load_config().get("unsupervised", {}).get(
            "min_cliff_delta", DEFAULT_MIN_CLIFF_DELTA
        )
    except (AttributeError, KeyError, TypeError, ValueError, OSError):  # fail-closed
        warnings.warn(
            "Could not read unsupervised.min_cliff_delta; "
            f"using default {DEFAULT_MIN_CLIFF_DELTA}.",
            UserWarning,
            stacklevel=2,
        )
        return DEFAULT_MIN_CLIFF_DELTA
    return float(value)


def add_wind_circular(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``wd_sin``/``wd_cos`` derived from ``wind_direction_10m``.

    Thin wrapper over :class:`AddWindCircular`; the raw circular column
    stays out of every feature set (plan §4, rule 2).
    """
    out = df.copy()
    derived = AddWindCircular().fit_transform(out[["wind_direction_10m"]])
    out[["wd_sin", "wd_cos"]] = derived
    return out


def apply_log_transforms(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Apply ``log1p`` to right-skewed columns (plan §5).

    Missing columns warn and are skipped (fail-closed, repo style).
    """
    out = df.copy()
    for col in LOG_COLS if columns is None else columns:
        if col not in out.columns:
            warnings.warn(
                f"log1p skipped: {col!r} not in frame.", UserWarning, stacklevel=2
            )
            continue
        out[col] = np.log1p(out[col].clip(lower=0))
    return out


def build_clc_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 8 CLC neighbour directions into environment features.

    Returns ``pct_vecinos_igual_centro`` (share of neighbours matching the
    centre pixel), ``n_clases_distintas_vecinos`` (1–8) and one
    ``freq_vecina_<clase>`` column per observed class. Missing neighbour
    columns warn and yield NaN (fail-closed).
    """
    missing = [c for c in ["clc_class", *CLC_NEIGHBOURS] if c not in df.columns]
    env = pd.DataFrame(index=df.index)
    if missing:
        warnings.warn(
            f"CLC environment incomplete, missing: {missing}.",
            UserWarning,
            stacklevel=2,
        )
        env["pct_vecinos_igual_centro"] = np.nan
        env["n_clases_distintas_vecinos"] = np.nan
        return env
    neigh = df[CLC_NEIGHBOURS]
    centre = df["clc_class"]
    env["pct_vecinos_igual_centro"] = neigh.eq(centre, axis=0).mean(axis=1)
    env["n_clases_distintas_vecinos"] = neigh.nunique(axis=1)
    for cls in sorted(neigh.stack().dropna().unique().tolist()):
        env[f"{FREQ_VECINA_PREFIX}{cls}"] = (neigh == cls).mean(axis=1)
    return env


def clc_environment_columns(env: pd.DataFrame) -> list[str]:
    """List the derived CLC-environment columns of a built frame."""
    return [
        c
        for c in env.columns
        if c in CLC_ENV_FIXED or c.startswith(FREQ_VECINA_PREFIX)
    ]


def assert_feature_sets_clean(*sets: list[str]) -> None:
    """Fail closed when a satellite column leaks into an input set.

    Satellite readings describe the sensor, not the cause (plan §4, rule 1):
    their presence in inputs is a bug, so this raises instead of warning.
    """
    for columns in sets:
        leaked = [c for c in columns if c in SATELLITE_BLOCKLIST]
        if leaked:
            raise ValueError(
                f"Satellite columns in unsupervised inputs: {leaked}. "
                "They belong to PERFILADO, never to ENTRADA_*."
            )


def resolve_set(df: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    """Select an input set, warning (not raising) on absent columns.

    Duplicate names collapse (first wins): set C lists the CLC fixed
    columns and the built environment, which share two names.
    """
    columns = list(dict.fromkeys(columns))
    absent = [c for c in columns if c not in df.columns]
    if absent:
        warnings.warn(
            f"{name}: absent columns dropped: {absent}.", UserWarning, stacklevel=2
        )
    return df[[c for c in columns if c in df.columns]].copy()


# ---------------------------------------------------------------------------
# Block 03b: clustering helpers (plan §6.1). The notebook measures (all k),
# 03e decides. Everything here is deterministic given ``seed``.
# ---------------------------------------------------------------------------

#: Shared seed (the project one).
CLUSTERING_SEED = 42

#: k sweep for KMeans/GMM (plan §6.1).
K_RANGE: list[int] = list(range(2, 13))

#: Ward sample size (dendrogram estimate, plan §6.1).
WARD_SAMPLE_N = 5_000

#: Silhouette is O(n²) memory: score it on a sample, the rest on full data.
SILHOUETTE_SAMPLE_N = 5_000

#: Ordered time-of-day levels: ordinal encoding is meaningful (night →
#: morning → heat peak → night), unlike one-hot which would explode k-means
#: distances on a single categorical.
FRANJA_ORDER: list[str] = ["Madrugada", "Mañana", "Tarde_Pico_Calor", "Noche"]

#: Nominal inputs excluded from distances; profiles/validation only
#: (plan §5: `clc_class` enters only methods that accept it — none yet).
NOMINAL_EXCLUDED: list[str] = ["clc_class"]

#: Stable row key for the whole 03 block: born in 02a as the row order of
#: `wildfire_data_cleaned.csv`, first column of `lags_index.csv` and of every
#: 03a output. Never a model/clustering input: excluded from distances.
ID_COL = "detection_id"


def attach_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert :data:`ID_COL` (0..n-1) as the first column (fail-closed)."""
    if ID_COL in frame.columns:
        raise ValueError(f"{ID_COL!r} already present: refusing to overwrite the key")
    out = frame.copy()
    out.insert(0, ID_COL, range(len(out)))
    return out


def check_ids(*frames: pd.DataFrame) -> np.ndarray:
    """Assert all frames share the identical :data:`ID_COL` key, return it.

    Replaces positional trust with a key check: same lengths are not enough,
    a re-sort must fail here instead of silently corrupting a join.
    """
    if not frames:
        raise ValueError("Need at least one frame")
    ids = []
    for i, frame in enumerate(frames):
        if ID_COL not in frame.columns:
            raise KeyError(f"Frame {i} carries no {ID_COL!r}")
        ids.append(np.asarray(frame[ID_COL]))
    first = ids[0]
    if any(not np.array_equal(other, first) for other in ids[1:]):
        raise ValueError("detection_id mismatch: frames are not in the same order")
    return first


def encode_franja_horaria(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode ``franja_horaria`` following :data:`FRANJA_ORDER`.

    Unknown levels warn and become NaN (fail-closed).
    """
    out = df.copy()
    if TIME_COL not in out.columns:
        warnings.warn(
            f"{TIME_COL!r} not in frame; nothing encoded.", UserWarning, stacklevel=2
        )
        return out
    codes = pd.Categorical(out[TIME_COL], categories=FRANJA_ORDER, ordered=True).codes
    unknown = set(out[TIME_COL].dropna().unique()) - set(FRANJA_ORDER)
    if unknown:
        warnings.warn(
            f"Unknown {TIME_COL} levels -> NaN: {sorted(unknown)}.",
            UserWarning,
            stacklevel=2,
        )
    out[TIME_COL] = codes.astype(float)
    out.loc[out[TIME_COL] < 0, TIME_COL] = np.nan
    return out


def distance_columns(columns: list[str]) -> list[str]:
    """Input columns usable in distance-based methods (drop nominals + key)."""
    excluded = set(NOMINAL_EXCLUDED) | {ID_COL}
    return [c for c in columns if c not in excluded]


def sample_frame(df: pd.DataFrame, n: int, seed: int = CLUSTERING_SEED) -> pd.DataFrame:
    """Deterministic uniform sample (Ward/silhouette scale control)."""
    return df.sample(n=min(n, len(df)), random_state=seed)


def fit_kmeans_range(
    X: pd.DataFrame,
    k_range: list[int] | None = None,
    seed: int = CLUSTERING_SEED,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """KMeans over ``k_range`` + internal metrics per k.

    Inertia/Calinski-Harabasz/Davies-Bouldin on full data; silhouette on a
    deterministic sample (memory). Returns (labels per k, metrics table).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    k_range = list(K_RANGE if k_range is None else k_range)
    values = X.to_numpy(dtype=float)
    sample_idx = sample_frame(
        pd.DataFrame(values), SILHOUETTE_SAMPLE_N, seed
    ).index.to_numpy()
    labels: dict[int, np.ndarray] = {}
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(values)
        lab = model.labels_
        labels[k] = lab
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(values[sample_idx], lab[sample_idx])),
                "calinski_harabasz": float(calinski_harabasz_score(values, lab)),
                "davies_bouldin": float(davies_bouldin_score(values, lab)),
            }
        )
    return labels, pd.DataFrame(rows).set_index("k")


def fit_gmm_range(
    X: pd.DataFrame,
    k_range: list[int] | None = None,
    covariance_types: tuple[str, ...] = ("full", "diag"),
    seed: int = CLUSTERING_SEED,
) -> dict[tuple[int, str], dict]:
    """GMM over (k, covariance) with BIC + soft membership (plan §6.1)."""
    from sklearn.mixture import GaussianMixture

    k_range = list(K_RANGE if k_range is None else k_range)
    values = X.to_numpy(dtype=float)
    results: dict[tuple[int, str], dict] = {}
    for k in k_range:
        for cov in covariance_types:
            model = GaussianMixture(
                n_components=k, covariance_type=cov, random_state=seed
            ).fit(values)
            results[(k, cov)] = {
                "labels": model.predict(values),
                "proba": model.predict_proba(values),
                "bic": float(model.bic(values)),
            }
    return results


def gmm_bic_table(results: dict[tuple[int, str], dict]) -> pd.DataFrame:
    """Pivot BIC per (k, covariance) for the k-by-BIC decision (03e)."""
    rows = [{"k": k, "covariance": cov, "bic": r["bic"]} for (k, cov), r in results.items()]
    return pd.DataFrame(rows).pivot(index="k", columns="covariance", values="bic")


def ward_linkage(X: pd.DataFrame) -> np.ndarray:
    """Ward linkage matrix on a (sampled) frame (plan §6.1, dendrogram)."""
    from scipy.cluster.hierarchy import linkage

    return linkage(X.to_numpy(dtype=float), method="ward")


def cliffs_delta(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    """Cliff's |δ|-signed effect size between two independent samples.

    (dominance rate: P(a > b) − P(a < b)). Rank-based via ``searchsorted``:
    O((n+m) log m), exact ties handled — identical values to the naive
    O(n·m) formulation, fast enough for 30 k-row group pairs.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    x = x[~np.isnan(x)]
    y = np.sort(y[~np.isnan(y)])
    if x.size == 0 or y.size == 0:
        return float("nan")
    smaller = (y.size - np.searchsorted(y, x, side="right")).sum()  # #(yj > xi)
    greater = np.searchsorted(y, x, side="left").sum()  # #(yj < xi)
    return float((greater - smaller) / (x.size * y.size))


def max_pairwise_cliffs_delta(
    values: np.ndarray | pd.Series, labels: np.ndarray | pd.Series
) -> tuple[float, tuple]:
    """Max |Cliff's δ| over group pairs (severity separation, plan §7B)."""
    vals = np.asarray(values, dtype=float).ravel()
    labs = np.asarray(labels).ravel()
    groups = sorted(set(labs.tolist()))
    best = (0.0, (None, None))
    for i, g in enumerate(groups):
        for h in groups[i + 1 :]:
            delta = cliffs_delta(vals[labs == g], vals[labs == h])
            if abs(delta) > abs(best[0]):
                best = (delta, (g, h))
    return best


def negative_control_partitions(perfilado: pd.DataFrame) -> dict[str, np.ndarray]:
    """Trivial month-only / CCAA-only partitions (plan §6.1, §7B).

    Deterministic baselines the candidate typology must beat: if it cannot
    separate severity better than the calendar, it rediscovered it.
    """
    controls = {}
    for name, col in (("month", "month"), ("ccaa", "ccaa")):
        if col not in perfilado.columns:
            warnings.warn(
                f"Negative control {name!r} skipped: {col!r} absent.",
                UserWarning,
                stacklevel=2,
            )
            continue
        controls[name] = pd.factorize(perfilado[col])[0]
    return controls


# ---------------------------------------------------------------------------
# Block 03c: anomaly helpers (plan §6.2). Scores are higher-is-rarer in all
# three detectors so flags and plots share one convention. Deterministic
# given ``seed``.
# ---------------------------------------------------------------------------

#: Votes needed for the consensus flag (plan §6.2: agreement of ≥2 of 3).
CONSENSUS_MIN_VOTES = 2

#: Default anomaly rate (top percentile of the score distribution, plan §6.2).
ANOMALY_PCT = 2.0


def fit_isolation_forest(
    X: pd.DataFrame, seed: int = CLUSTERING_SEED
) -> np.ndarray:
    """Isolation-Forest rarity scores (higher = rarer)."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(random_state=seed).fit(X.to_numpy(dtype=float))
    return -model.decision_function(X.to_numpy(dtype=float))


def fit_lof(X: pd.DataFrame) -> np.ndarray:
    """Local-Outlier-Factor rarity scores (higher = rarer).

    Local by construction: flags points sparse *relative to their
    neighbourhood*, e.g. a middling fire where all fires are small.
    """
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor().fit(X.to_numpy(dtype=float))
    return -model.negative_outlier_factor_


def robust_mahalanobis(
    X: pd.DataFrame, seed: int = CLUSTERING_SEED
) -> np.ndarray:
    """Robust Mahalanobis distances via MinCovDet (higher = rarer).

    Global and elliptical: flags points far from the robust centre after
    accounting for correlations (complements the local view of LOF).
    """
    from sklearn.covariance import MinCovDet

    values = X.to_numpy(dtype=float)
    return MinCovDet(random_state=seed).fit(values).mahalanobis(values)


def percentile_flag(scores: np.ndarray | pd.Series, pct: float = ANOMALY_PCT) -> np.ndarray:
    """Flag the top ``pct`` % of a rarity score (never an eyeballed rate)."""
    s = np.asarray(scores, dtype=float).ravel()
    if not 0 < pct < 100:
        raise ValueError(f"pct must be in (0, 100), got {pct}.")
    threshold = np.nanquantile(s, 1.0 - pct / 100.0)
    return s >= threshold


def consensus_flag(*flags: np.ndarray, min_votes: int = CONSENSUS_MIN_VOTES) -> np.ndarray:
    """Keep rows flagged by at least ``min_votes`` detectors."""
    stacked = np.column_stack([np.asarray(f, dtype=bool).ravel() for f in flags])
    return stacked.sum(axis=1) >= min_votes


def imputed_blh_mask(df: pd.DataFrame, col: str = "boundary_layer_height") -> pd.Series:
    """Flag rows sitting exactly at the column median (plan §3 artefact hunt).

    The 16 % BLH gap was median-imputed in notebook 01, so rows exactly at
    the median are likely imputed, not measured. Missing column -> all False
    with a warning (fail-closed).
    """
    if col not in df.columns:
        warnings.warn(
            f"{col!r} absent; no BLH-imputed rows flagged.", UserWarning, stacklevel=2
        )
        return pd.Series(False, index=df.index)
    median = float(df[col].median())
    return df[col] == median


def weather_pc_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Project rows onto the weather-PCA space (PC1..PCk, plan §4.3 visual).

    Thin wrapper over :func:`fit_weather_pca`: same 91.5 % / 7-PC space as
    02b, refit here descriptively for colouring anomaly maps (never an input
    to another model without refit-on-train).
    """
    from wildfire.processing.pca_weather import fit_weather_pca, pca_input_columns

    missing = set(pca_input_columns()) - set(df.columns)
    if missing:
        raise KeyError(f"Missing PCA input columns: {sorted(missing)}.")
    return fit_weather_pca(df).scores
