"""Wide-margin exceedance calibration (Plan D, route C — provisional).

The occurrence model produces a **ranking**, not a probability (prevalence is
set by the negative sampling). This module turns that ranking into an honest
*exceedance* probability at a **wide spatial margin** ``R``:

    ``P(>=1 detection within R km of a point on day d)``

Point-level calibration is deliberately avoided (near-zero, unestimable
prevalence): at a wide margin the event is common enough to be measured. The
recipe is the one agreed for block 04:

1. **Label by margin** — a validation point is positive-at-``R`` when a
   same-day detection lies within ``R`` km (:func:`margin_positive_labels`).
2. **Isotonic fit** — map the model score to a probability with
   :class:`~sklearn.isotonic.IsotonicRegression`, fit on the held-out scores of
   a *temporally* later year (2023 train -> 2024 calibrate).
3. **Cross-fit** — reliability is reported from cross-fitted predictions
   (:func:`cross_fitted_isotonic`) so the calibrator is not scored on its own
   fit; the persisted calibrator is refit on the full calibration year.
4. **Prior correction** — rescale to the *measured* base rate
   (:func:`exceedance_base_rate`, estimated on the full background pool, not
   the 1:``ratio`` training sample) with :func:`prior_correct`.
5. **Margins** — sweep ``R`` in ``EXCEEDANCE_MARGINS_KM`` {5, 10, 20} km;
   ``eventos.csv`` event ids keep the effective sample size honest.

Everything here is **experimental**: the probabilities carry the provisional
background pool and the sampled (not uniform) evaluation distribution as
caveats. See ``docs/calibration-C.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    brier_score_loss,
    log_loss,
)
from sklearn.neighbors import BallTree  # type: ignore[import-untyped]

#: Mean Earth radius (km) for the haversine margin queries.
_EARTH_RADIUS_KM = 6371.0088

#: Margin sweep (km) for the exceedance probability (Plan D, route C).
EXCEEDANCE_MARGINS_KM: tuple[float, ...] = (5.0, 10.0, 20.0)

#: Probability clip for odds-based prior correction (avoids log/0).
_EPS = 1e-6


def margin_positive_labels(
    points: pd.DataFrame,
    detections: pd.DataFrame,
    margin_km: float,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "acq_date",
) -> np.ndarray:
    """Boolean ``>=1 same-day detection within ``margin_km`` of each point``.

    Date-partitioned haversine ``BallTree`` over the detections; rows with
    unparseable dates or non-finite coordinates are ``False`` (never crash).
    A detection is trivially within 0 km of itself, so positives come back
    ``True`` for any ``margin_km``.
    """
    if margin_km < 0:
        raise ValueError(f"margin_km must be >= 0, got {margin_km!r}")
    required = (lat_col, lon_col, date_col)
    for name, frame in (("points", points), ("detections", detections)):
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise KeyError(f"{name} missing columns for margin labeling: {missing}")

    det = detections[list(required)].dropna(subset=[lat_col, lon_col])
    det_dates = pd.to_datetime(det[date_col], errors="coerce")
    keep = det_dates.notna().to_numpy()
    det_coords = np.radians(det[[lat_col, lon_col]].to_numpy(dtype=float))[keep]
    det_day = det_dates.dt.strftime("%Y-%m-%d").to_numpy()[keep]
    by_day: dict[str, np.ndarray] = {
        d: det_coords[det_day == d] for d in np.unique(det_day)
    }

    pts = points[list(required)]
    p_dates = pd.to_datetime(pts[date_col], errors="coerce")
    p_coords = np.radians(pts[[lat_col, lon_col]].to_numpy(dtype=float))
    p_day = p_dates.dt.strftime("%Y-%m-%d").to_numpy()
    finite = np.isfinite(p_coords).all(axis=1)

    radius = float(margin_km) / _EARTH_RADIUS_KM
    hits = np.zeros(len(points), dtype=bool)
    for d in np.unique(p_day):
        day_det = by_day.get(d)
        if day_det is None:
            continue
        rows = np.flatnonzero((p_day == d) & finite)
        if rows.size == 0:
            continue
        tree = BallTree(day_det, metric="haversine")
        distance, _ = tree.query(p_coords[rows], k=1)
        hits[rows] = distance[:, 0] <= radius
    return hits


def fit_isotonic(
    scores: np.ndarray | pd.Series, y: np.ndarray | pd.Series
) -> IsotonicRegression:
    """Fit an isotonic score->probability map (both classes required)."""
    s = np.asarray(scores, dtype=float)
    labels = np.asarray(y)
    if s.shape[0] != labels.shape[0]:
        raise ValueError("scores and y must have the same length")
    if not np.isfinite(s).all():
        raise ValueError("fit_isotonic needs finite scores")
    if np.unique(labels).size != 2:
        raise ValueError("fit_isotonic needs both classes present")
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(s, labels.astype(float))
    return model


def apply_isotonic(
    model: IsotonicRegression, scores: np.ndarray | pd.Series
) -> np.ndarray:
    """Apply a fitted isotonic map to ``scores`` (clipped to [0, 1])."""
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("apply_isotonic needs finite scores")
    return np.clip(model.predict(s), 0.0, 1.0)


def prior_correct(
    probabilities: np.ndarray | pd.Series,
    *,
    source_prevalence: float,
    target_prevalence: float,
) -> np.ndarray:
    """Rescale probabilities from a sampled prevalence to a measured one.

    Standard odds correction: ``odds_target = odds_source *
    (t/(1-t)) / (s/(1-s))``. Both prevalences must be in ``(0, 1)``; the
    calibrated probabilities are clipped off 0/1 first.
    """
    for name, prev in (("source", source_prevalence), ("target", target_prevalence)):
        if not 0.0 < float(prev) < 1.0:
            raise ValueError(f"{name}_prevalence must be in (0, 1), got {prev!r}")
    p = np.clip(np.asarray(probabilities, dtype=float), _EPS, 1.0 - _EPS)
    factor = (target_prevalence / (1.0 - target_prevalence)) / (
        source_prevalence / (1.0 - source_prevalence)
    )
    odds = (p / (1.0 - p)) * factor
    return odds / (1.0 + odds)


def reliability_curve(
    y: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Binned observed-vs-predicted frequency table (calibration curve).

    ``strategy="quantile"`` makes equal-count bins (rank-based, so duplicate
    scores never break ``qcut``); ``"uniform"`` uses fixed [0, 1] edges.
    """
    labels = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if labels.shape[0] != p.shape[0]:
        raise ValueError("y and probabilities must have the same length")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins!r}")
    if strategy not in ("quantile", "uniform"):
        raise ValueError(f"unknown strategy: {strategy!r}")
    if strategy == "quantile":
        ranks = pd.Series(p).rank(method="first")
        binned = pd.qcut(ranks, q=min(n_bins, max(1, len(p))), labels=False)
    else:
        binned = pd.cut(pd.Series(p), bins=n_bins, labels=False, include_lowest=True)
    frame = pd.DataFrame({"bin": np.asarray(binned), "y": labels, "p": p})
    rows = []
    for _, group in frame.groupby("bin", sort=True):
        rows.append(
            {
                "bin_lower": float(group["p"].min()),
                "bin_upper": float(group["p"].max()),
                "mean_predicted": float(group["p"].mean()),
                "observed_freq": float(group["y"].mean()),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows)


def calibration_metrics(
    y: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
) -> dict:
    """Brier score, log-loss and expected calibration error (ECE)."""
    labels = np.asarray(y)
    p = np.asarray(probabilities, dtype=float)
    if labels.shape[0] != p.shape[0]:
        raise ValueError("y and probabilities must have the same length")
    if np.unique(labels).size != 2:
        raise ValueError("calibration_metrics needs both classes present")
    curve = reliability_curve(labels, p, n_bins=n_bins, strategy="quantile")
    n_total = int(curve["n"].sum())
    ece = float(
        (
            curve["n"]
            / n_total
            * (curve["mean_predicted"] - curve["observed_freq"]).abs()
        ).sum()
    )
    return {
        "brier": float(brier_score_loss(labels, np.clip(p, _EPS, 1.0 - _EPS))),
        "log_loss": float(log_loss(labels, np.clip(p, _EPS, 1.0 - _EPS))),
        "ece": ece,
    }


def dedupe_events(
    detections: pd.DataFrame,
    events: pd.DataFrame,
    *,
    detection_col: str = "detection_id",
    event_col: str = "event_id",
) -> pd.DataFrame:
    """Merge the event id and keep one detection per event (first by ``detection_id``).

    FIRMS overpasses fire several times per event; counting unique events
    (not detections) keeps reported sample sizes honest.
    """
    missing = [c for c in (detection_col, event_col) if c not in events.columns]
    if missing:
        raise KeyError(f"events missing columns: {missing}")
    if detection_col not in detections.columns:
        raise KeyError(f"detections missing column: {detection_col}")
    merged = detections.merge(
        events[[detection_col, event_col]], on=detection_col, how="left"
    )
    merged = merged.sort_values(detection_col, kind="stable")
    return merged.drop_duplicates(subset=[event_col], keep="first").reset_index(
        drop=True
    )


def exceedance_base_rate(
    background: pd.DataFrame,
    detections: pd.DataFrame,
    margin_km: float,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "acq_date",
) -> dict:
    """Measured ``P(>=1 detection within ``margin_km``)`` on the background pool.

    The background rows are a CCAA-day sample of the point-day universe, so
    their margin-positive fraction is a direct (model-free) base-rate estimate.
    The CI resamples whole **days** (block bootstrap) to respect the strong
    day-level clustering.
    """
    hits = margin_positive_labels(
        background,
        detections,
        margin_km,
        lat_col=lat_col,
        lon_col=lon_col,
        date_col=date_col,
    )
    days = pd.to_datetime(background[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    rate = float(hits.mean()) if len(hits) else float("nan")

    rng = np.random.default_rng(seed)
    unique_days = pd.unique(days)
    by_day = {d: hits[(days == d).to_numpy()] for d in unique_days}
    boot = []
    if len(unique_days) > 1:
        for _ in range(n_boot):
            picked = rng.choice(unique_days, size=len(unique_days), replace=True)
            boot.append(float(np.concatenate([by_day[d] for d in picked]).mean()))
    ci_low = float(np.percentile(boot, 2.5)) if boot else rate
    ci_high = float(np.percentile(boot, 97.5)) if boot else rate
    return {
        "rate": rate,
        "n_points": len(hits),
        "n_positive": int(hits.sum()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def cross_fitted_isotonic(
    scores: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    n_splits: int = 5,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Cross-fitted isotonic probabilities + a final calibrator.

    ``groups`` (e.g. the day) are split into ``n_splits`` contiguous blocks;
    each fold's probabilities come from a calibrator fit on the other blocks,
    so the returned predictions are not scored on their own fit. The second
    return value is refit on **all** rows (the deployment calibrator).
    """
    s = np.asarray(scores, dtype=float)
    labels = np.asarray(y, dtype=float)
    g = np.asarray(groups)
    if not (s.shape[0] == labels.shape[0] == g.shape[0]):
        raise ValueError("scores, y and groups must have the same length")
    if np.unique(labels).size != 2:
        raise ValueError("cross_fitted_isotonic needs both classes present")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits!r}")

    unique_groups = pd.unique(g)
    blocks = np.array_split(unique_groups, min(n_splits, len(unique_groups)))
    fold_of = {}
    for i, block in enumerate(blocks):
        for value in block:
            fold_of[value] = i
    fold = np.array([fold_of[value] for value in g])

    pooled = np.full(s.shape[0], np.nan)
    for i in range(len(blocks)):
        test = fold == i
        train = ~test
        if not test.any():
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(s[train], labels[train])
        pooled[test] = np.clip(model.predict(s[test]), 0.0, 1.0)

    final = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    final.fit(s, labels)
    return pooled, final


@dataclass
class MarginCalibration:
    """Per-margin calibration result (see :func:`calibrate_exceedance`)."""

    margin_km: float
    calibrator: IsotonicRegression
    source_prevalence: float
    target_prevalence: float
    prior_corrected_mean: float
    metrics: dict
    reliability: pd.DataFrame
    n_points: int
    n_margin_positive: int


def calibrate_exceedance(
    validation: pd.DataFrame,
    detections: pd.DataFrame,
    background: pd.DataFrame,
    *,
    margins: tuple[float, ...] = EXCEEDANCE_MARGINS_KM,
    score_col: str = "score",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    date_col: str = "acq_date",
    n_splits: int = 5,
    n_bins: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[float, MarginCalibration]:
    """Calibrate the ranking into an exceedance probability per margin.

    ``validation`` must carry ``score_col`` plus the coordinates/date used for
    margin labeling and cross-fitting; ``detections`` is the same-day proximity
    source; ``background`` (the full pool) sets the measured base rate. The
    calibration year should be *later* than the training year (2024 vs 2023).
    """
    for col in (score_col, lat_col, lon_col, date_col):
        if col not in validation.columns:
            raise KeyError(f"validation lacks column {col!r}")
    scores = validation[score_col].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("validation scores must be finite")
    days = (
        pd.to_datetime(validation[date_col], errors="coerce")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )

    results: dict[float, MarginCalibration] = {}
    for margin in margins:
        y = margin_positive_labels(
            validation,
            detections,
            margin,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
        )
        if np.unique(y).size != 2:
            raise ValueError(
                f"margin {margin} km has a single class on the validation set "
                "(widen the range or grow the frame)"
            )
        pooled, calibrator = cross_fitted_isotonic(scores, y, days, n_splits=n_splits)
        source_prev = float(y.mean())
        base = exceedance_base_rate(
            background,
            detections,
            margin,
            n_boot=n_boot,
            seed=seed,
            lat_col=lat_col,
            lon_col=lon_col,
            date_col=date_col,
        )
        corrected = prior_correct(
            pooled, source_prevalence=source_prev, target_prevalence=base["rate"]
        )
        results[float(margin)] = MarginCalibration(
            margin_km=float(margin),
            calibrator=calibrator,
            source_prevalence=source_prev,
            target_prevalence=base["rate"],
            prior_corrected_mean=float(corrected.mean()),
            metrics=calibration_metrics(y, pooled, n_bins=n_bins),
            reliability=reliability_curve(y, pooled, n_bins=n_bins),
            n_points=len(y),
            n_margin_positive=int(y.sum()),
        )
    return results


def exceedance_summary(results: dict[float, MarginCalibration]) -> pd.DataFrame:
    """Tidy one-row-per-margin table from :func:`calibrate_exceedance`."""
    rows = []
    for margin in sorted(results):
        result = results[margin]
        rows.append(
            {
                "margin_km": result.margin_km,
                "n_points": result.n_points,
                "n_margin_positive": result.n_margin_positive,
                "source_prevalence": result.source_prevalence,
                "base_rate": result.target_prevalence,
                "prior_corrected_mean": result.prior_corrected_mean,
                "brier": result.metrics["brier"],
                "log_loss": result.metrics["log_loss"],
                "ece": result.metrics["ece"],
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "EXCEEDANCE_MARGINS_KM",
    "MarginCalibration",
    "apply_isotonic",
    "calibrate_exceedance",
    "calibration_metrics",
    "cross_fitted_isotonic",
    "dedupe_events",
    "exceedance_base_rate",
    "exceedance_summary",
    "fit_isotonic",
    "margin_positive_labels",
    "prior_correct",
    "reliability_curve",
]
