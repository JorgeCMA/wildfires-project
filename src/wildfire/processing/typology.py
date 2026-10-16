"""Typology validation (block 03e): stability (§7A) + utility (§7B).

A conditions typology is accepted only if it passes both: the structure must
exist (stable under resampling, space and time) and it must help prediction
(separates severity beyond what the calendar or the map already do).
Implements ``planning/plan_unsupervised_next_implementations.md`` §2.
Notebook 03e displays and orchestrates; the units that can break silently
live here, tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from wildfire.processing.unsupervised import (
    CLUSTERING_SEED,
    load_min_cliff_delta,
    max_pairwise_cliffs_delta,
)

#: Resample design for the stability check (plan §7A).
STABILITY_RESAMPLES = 30
STABILITY_FRAC = 0.8

#: Acceptance thresholds (plan §7A/§7B).
STABILITY_ARI_MIN = 0.7
PREPROCESSING_ARI_MIN = 0.6
NMI_TRIVIALITY_MAX = 0.5
GMM_OVERLAP_POSTERIOR = 0.8
#: Spatial gate: every refit must keep all k groups AND the worst matched
#: profile correlation must clear this (judgment bar: the plan fixes no
#: number, only "no group disappears"; correlation is reported regardless).
SPATIAL_CORR_MIN = 0.5
#: Overlap gate: above this share of diffuse posteriors the hard labels stop
#: meaning regimes (judgment bar, reported regardless).
GMM_OVERLAP_MAX = 0.25


def _standardize(X: np.ndarray, robust: bool = False) -> np.ndarray:
    scaler = RobustScaler() if robust else StandardScaler()
    return scaler.fit_transform(np.asarray(X, dtype=float))


def fit_labels(X_scaled: np.ndarray, k: int, seed: int = CLUSTERING_SEED) -> np.ndarray:
    """KMeans hard labels on an already-scaled matrix (deterministic)."""
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k!r}")
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(
        np.asarray(X_scaled, dtype=float)
    )


def resample_stability(
    X: pd.DataFrame | np.ndarray,
    k: int,
    n_resamples: int = STABILITY_RESAMPLES,
    frac: float = STABILITY_FRAC,
    seed: int = CLUSTERING_SEED,
) -> dict:
    """ARI of refits on random subsets vs full-data labels (existence check).

    Each resample fits its own scaler (rule 4.3: no scaler reuse); ARI is
    computed on the resampled rows only. Pass bar: mean ARI > 0.7.
    """
    mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    n = len(mat)
    if n == 0:
        raise ValueError("Empty matrix")
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac!r}")
    full = fit_labels(_standardize(mat), k, seed)
    rng = np.random.default_rng(seed)
    m = max(k + 1, int(n * frac))
    aris = []
    for _ in range(n_resamples):
        idx = np.sort(rng.choice(n, size=m, replace=False))
        sub = fit_labels(_standardize(mat[idx]), k, seed)
        aris.append(adjusted_rand_score(full[idx], sub))
    aris = np.asarray(aris)
    return {"mean_ari": float(aris.mean()), "min_ari": float(aris.min()), "aris": aris}


def _match_profile_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Correlation between two group-profile tables with matched groups.

    KMeans labels are arbitrary per fit, so rows are matched first (Hungarian
    assignment on centroid distances); correlating unmatched tables would
    punish identical structures for relabelling. Returns NaN when the tables
    have different numbers of groups (a group appeared or vanished).
    """
    if len(a) != len(b):
        return float("nan")
    cost = np.linalg.norm(
        a.sort_index().to_numpy()[:, None, :] - b.sort_index().to_numpy()[None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(cost)
    matched_a = a.sort_index().to_numpy()[rows].ravel()
    matched_b = b.sort_index().to_numpy()[cols].ravel()
    if np.std(matched_a) == 0 or np.std(matched_b) == 0:
        return float("nan")
    return float(np.corrcoef(matched_a, matched_b)[0, 1])


def leave_one_group_out(
    X: pd.DataFrame | np.ndarray,
    groups: pd.Series | np.ndarray,
    k: int,
    seed: int = CLUSTERING_SEED,
) -> pd.DataFrame:
    """Refit without each group; does every group survive (spatial check)?

    Per held-out group: how many of the k groups the refit still finds, the
    matched-profile correlation with the full-data profiles, and (diagnostic
    only) where the held-out rows concentrate. A group that vanishes is a
    regional artefact, not a typology (plan §7A).
    """
    mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    groups = pd.Series(np.asarray(groups)).reset_index(drop=True)
    full_profiles = (
        pd.DataFrame(mat).assign(grupo=fit_labels(_standardize(mat), k, seed)).groupby("grupo").mean()
    )
    rows = []
    for grp in sorted(groups.unique().tolist()):
        held = groups[groups == grp].index.to_numpy()
        rest = groups[groups != grp].index.to_numpy()
        if len(rest) < k:
            raise ValueError(f"Too few rows outside group {grp!r}")
        scaler = StandardScaler().fit(mat[rest])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(mat[rest]))
        refit_profiles = (
            pd.DataFrame(mat[rest]).assign(grupo=km.labels_).groupby("grupo").mean()
        )
        assigned = km.predict(scaler.transform(mat[held]))
        _, counts = np.unique(assigned, return_counts=True)
        rows.append(
            {
                "group": grp,
                "n_heldout": len(held),
                "groups_found": len(set(km.labels_.tolist())),
                "profile_corr": _match_profile_corr(full_profiles, refit_profiles),
                "assigned_top_frac": float(counts.max() / len(held)),
            }
        )
    return pd.DataFrame(rows)


def cross_year_profiles(
    X: pd.DataFrame | np.ndarray,
    year: pd.Series | np.ndarray,
    k: int,
    seed: int = CLUSTERING_SEED,
) -> dict:
    """Fit on each year, assign the other: do group profiles replicate?

    Returns per-direction group means plus the correlation between the two
    directions' profile tables (high = the same regimes both years).
    """
    mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    year = pd.Series(np.asarray(year)).reset_index(drop=True)
    years = sorted(year.unique().tolist())
    if len(years) != 2:
        raise ValueError(f"Need exactly 2 years, got {years}")
    profiles = {}
    for train_y, test_y in [(years[0], years[1]), (years[1], years[0])]:
        tr = (year == train_y).to_numpy()
        te = (year == test_y).to_numpy()
        scaler = StandardScaler().fit(mat[tr])
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(scaler.transform(mat[tr]))
        lab = km.predict(scaler.transform(mat[te]))
        profiles[f"{train_y}->{test_y}"] = pd.DataFrame(mat[te]).assign(grupo=lab).groupby("grupo").mean()
    keys = sorted(profiles)
    corr = _match_profile_corr(profiles[keys[0]], profiles[keys[1]])
    return {"profiles": profiles, "profile_corr": corr}


def pairwise_ari(labelings: dict[str, np.ndarray]) -> pd.DataFrame:
    """ARI matrix across preprocessing variants (robustness check).

    Pass bar: every pair > 0.6. Keys name the variants (e.g. standard/robust).
    """
    names = list(labelings.keys())
    if len(names) < 2:
        raise ValueError("Need at least 2 labelings")
    n = len(labelings[names[0]])
    if any(len(v) != n for v in labelings.values()):
        raise ValueError("All labelings must cover the same rows")
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b_ in names:
            table.loc[a, b_] = adjusted_rand_score(labelings[a], labelings[b_])
    return table


def severity_separation(
    labels: pd.Series | np.ndarray, frp: pd.Series | np.ndarray
) -> dict:
    """Kruskal on FRP across groups + max pairwise Cliff's |δ| (utility check).

    Pass bar: at least one pair with |δ| >= min_cliff_delta (config-anchored).
    """
    lab = pd.Series(np.asarray(labels)).reset_index(drop=True)
    val = pd.Series(np.asarray(frp, dtype=float)).reset_index(drop=True)
    if len(lab) != len(val):
        raise ValueError("length mismatch")
    stat, p = stats.kruskal(*[val[lab == g].to_numpy() for g in sorted(lab.unique())])
    delta, pair = max_pairwise_cliffs_delta(val, lab)
    threshold = load_min_cliff_delta()
    return {
        "kruskal_h": float(stat),
        "kruskal_p": float(p),
        "max_abs_delta": float(abs(delta)),
        "pair": pair,
        "threshold": float(threshold),
        "passes": bool(abs(delta) >= threshold),
    }


def noncoincidence_ari(
    candidate: pd.Series | np.ndarray,
    month_labels: pd.Series | np.ndarray,
    ccaa_labels: pd.Series | np.ndarray,
) -> dict:
    """ARI of the candidate vs the trivial partitions (must be low on both)."""
    cand = np.asarray(candidate)
    out = {
        "ari_vs_month": float(adjusted_rand_score(cand, np.asarray(month_labels))),
        "ari_vs_ccaa": float(adjusted_rand_score(cand, np.asarray(ccaa_labels))),
    }
    return out


def nontriviality_nmi(candidate: pd.Series | np.ndarray, **partitions: pd.Series | np.ndarray) -> dict:
    """Normalised MI vs calendar/geography variables (must all be <= 0.5)."""
    cand = np.asarray(candidate)
    out = {name: float(normalized_mutual_info_score(cand, np.asarray(col))) for name, col in partitions.items()}
    out["passes"] = bool(all(v <= NMI_TRIVIALITY_MAX for v in out.values()))
    return out


def gmm_overlap_mass(proba: pd.DataFrame | np.ndarray, posterior: float = GMM_OVERLAP_POSTERIOR) -> dict:
    """Share of rows below a max-posterior cutoff (regime-overlap quantifier)."""
    mat = proba.to_numpy(dtype=float) if isinstance(proba, pd.DataFrame) else np.asarray(proba, dtype=float)
    if len(mat) == 0:
        raise ValueError("Empty posterior matrix")
    if not np.allclose(mat.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Posterior rows must sum to 1")
    confident = mat.max(axis=1)
    frac = float((confident < posterior).mean())
    return {
        "frac_below": frac,
        "median_max_posterior": float(np.median(confident)),
        "passes": bool(frac < GMM_OVERLAP_MAX),
    }


def anchor_min_delta(candidate_max: float, control_maxes: dict[str, float]) -> dict:
    """Anchor the Cliff's δ bar just above the best trivial partition.

    The bar is the smallest 0.05 step strictly above the strongest control,
    so a future candidate passes only if it beats month AND ccaa (rounded-up
    grid keeps the config value stable instead of chasing decimals).
    """
    best_name = max(control_maxes, key=lambda k: control_maxes[k])
    best = float(control_maxes[best_name])
    bar = float(np.ceil((best + 1e-9) * 20.0) / 20.0)
    return {
        "best_control": best_name,
        "best_control_max": best,
        "candidate_max": float(candidate_max),
        "recommended_bar": bar,
        "candidate_beats_bar": bool(candidate_max >= bar),
    }


def validation_verdict(checks_7a: dict[str, bool], checks_7b: dict[str, bool]) -> str:
    """Final typology verdict: accepted iff every check passes.

    7A failure (structure does not exist) rejects outright; 7A green but 7B
    red (exists but utility unproven) is inconclusive, not accepted.
    """
    if not checks_7a or not checks_7b:
        raise ValueError("Both check blocks must be non-empty")
    if not all(checks_7a.values()):
        return "rejected"
    if not all(checks_7b.values()):
        return "inconclusive"
    return "accepted"


__all__ = [
    "GMM_OVERLAP_MAX",
    "GMM_OVERLAP_POSTERIOR",
    "NMI_TRIVIALITY_MAX",
    "PREPROCESSING_ARI_MIN",
    "SPATIAL_CORR_MIN",
    "STABILITY_ARI_MIN",
    "STABILITY_FRAC",
    "STABILITY_RESAMPLES",
    "anchor_min_delta",
    "cross_year_profiles",
    "fit_labels",
    "gmm_overlap_mass",
    "leave_one_group_out",
    "noncoincidence_ari",
    "nontriviality_nmi",
    "pairwise_ari",
    "resample_stability",
    "severity_separation",
    "validation_verdict",
]
