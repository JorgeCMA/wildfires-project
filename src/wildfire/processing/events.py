"""Spatio-temporal fire events (block 03d).

DBSCAN over ``(x_km, y_km, day_scaled)`` answers how many real fires stand
behind the detections and whether one event spans several conditions-regimes
(``planning/plan_unsupervised_next_implementations.md`` §1). Notebook 03d
displays and orchestrates; the units that can break silently live here, tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#: Geographic (degrees) and metric (metres) CRS. 3035 is equal-area over all
#: of Spain — UTM 30N does not cover Canarias, so it would distort DBSCAN
#: distances there (plan correction 2026-10-16).
CRS_GEO = "EPSG:4326"
CRS_METRIC = "EPSG:3035"

#: Anisotropic scaling: 1 day counts as this many km (default + sensitivity).
DEFAULT_DAY_KM = 7.5
DAY_KM_SENSITIVITY: tuple[float, ...] = (5.0, 7.5, 10.0)

#: A 2-detection group is a re-detection, not event structure (provisional).
DEFAULT_MIN_SAMPLES = 3

#: DBSCAN label for noise (isolated detections, e.g. agricultural burns).
NOISE_LABEL = -1

#: Set-D input columns consumed by :func:`build_event_matrix`.
EVENT_INPUT_COLS: list[str] = ["latitude", "longitude", "acq_date"]

#: Matrix columns produced by :func:`build_event_matrix` (DBSCAN space).
EVENT_MATRIX_COLS: list[str] = ["x_km", "y_km", "day", "day_scaled"]

#: Base columns of the per-event summary (PC means appended when given).
EVENT_SUMMARY_BASE: list[str] = [
    "event_id",
    "n",
    "first_day",
    "last_day",
    "duration_days",
    "x_centroid_km",
    "y_centroid_km",
    "x_span_km",
    "y_span_km",
    "ccaa_top",
    "ccaa_n",
    "frp_max",
]


def _project_lonlat(longitude: pd.Series, latitude: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Project lon/lat degrees to 3035 metres (``always_xy``: lon first)."""
    transformer = Transformer.from_crs(CRS_GEO, CRS_METRIC, always_xy=True)
    xs, ys = transformer.transform(
        np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float)
    )
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def build_event_matrix(df: pd.DataFrame, day_km: float = DEFAULT_DAY_KM) -> pd.DataFrame:
    """Build the DBSCAN space ``(x_km, y_km, day_scaled)`` (fail-closed).

    ``day`` is the integer day index from the earliest detection; ``day_scaled``
    stretches it so that 1 day counts as ``day_km`` kilometres (anisotropic
    scaling bridges FIRMS overpass gaps). The input index is preserved so
    labels join back positionally.
    """
    if day_km <= 0:
        raise ValueError(f"day_km must be > 0, got {day_km!r}")
    missing = [c for c in EVENT_INPUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing event input columns: {missing}")
    if len(df) == 0:
        raise ValueError("Empty frame: nothing to build events from")
    dates = pd.to_datetime(df["acq_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("acq_date has unparseable values")
    xs, ys = _project_lonlat(df["longitude"], df["latitude"])
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("Non-finite projected coordinates (bad lat/lon?)")
    day = (dates.dt.normalize() - dates.dt.normalize().min()).dt.days.to_numpy()
    out = pd.DataFrame(
        {
            "x_km": xs / 1000.0,
            "y_km": ys / 1000.0,
            "day": day.astype(int),
            "day_scaled": day.astype(float) * day_km,
        },
        index=df.index,
    )
    return out


def k_distance_curve(X: pd.DataFrame | np.ndarray, k: int) -> np.ndarray:
    """Sorted ascending distances to the k-th nearest neighbour (eps knee).

    ``X`` is the DBSCAN space (``x_km``, ``y_km``, ``day_scaled``); ``k``
    matches ``min_samples``. The notebook plots the curve and reads the knee.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if len(mat) <= k:
        raise ValueError(f"Need more than k={k} rows, got {len(mat)}")
    dists, _ = NearestNeighbors(n_neighbors=k + 1).fit(mat).kneighbors(mat)
    return np.sort(dists[:, -1])


def fit_dbscan_events(
    X: pd.DataFrame | np.ndarray,
    eps: float,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> np.ndarray:
    """DBSCAN labels over the event matrix; ``-1`` is noise (fail-closed)."""
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples!r}")
    mat = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
    if len(mat) == 0:
        raise ValueError("Empty matrix: nothing to cluster")
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mat)


def summarize_events(
    frame: pd.DataFrame,
    labels: np.ndarray,
    pc_cols: list[str] | None = None,
) -> pd.DataFrame:
    """One row per event (noise excluded): size, span, CCAA mix, max FRP.

    ``frame`` carries ``x_km``, ``y_km``, ``day``, ``ccaa``, ``frp``;
    ``labels`` are the aligned DBSCAN labels. ``pc_cols`` (e.g. set-A PCs)
    are averaged per event when given. All-noise input returns the empty
    frame with the full schema.
    """
    required = ["x_km", "y_km", "day", "ccaa", "frp"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing summary columns: {missing}")
    labels = np.asarray(labels)
    if len(labels) != len(frame):
        raise ValueError(f"labels ({len(labels)}) != frame rows ({len(frame)})")
    pc_cols = list(pc_cols or [])
    unknown = [c for c in pc_cols if c not in frame.columns]
    if unknown:
        raise KeyError(f"Unknown pc_cols: {unknown}")
    cols = [*EVENT_SUMMARY_BASE, *[f"{c}_mean" for c in pc_cols]]
    events = sorted(set(labels.tolist()) - {NOISE_LABEL})
    if not events:
        return pd.DataFrame(columns=cols)
    frame = frame.copy()
    frame["_event"] = labels
    rows = []
    for ev in events:
        sub = frame[frame["_event"] == ev]
        rows.append(
            {
                "event_id": ev,
                "n": len(sub),
                "first_day": int(sub["day"].min()),
                "last_day": int(sub["day"].max()),
                "duration_days": int(sub["day"].max() - sub["day"].min()) + 1,
                "x_centroid_km": float(sub["x_km"].mean()),
                "y_centroid_km": float(sub["y_km"].mean()),
                "x_span_km": float(sub["x_km"].max() - sub["x_km"].min()),
                "y_span_km": float(sub["y_km"].max() - sub["y_km"].min()),
                "ccaa_top": str(sub["ccaa"].mode().iat[0]),
                "ccaa_n": int(sub["ccaa"].nunique()),
                "frp_max": float(sub["frp"].max()),
                **{f"{c}_mean": float(sub[c].mean()) for c in pc_cols},
            }
        )
    return pd.DataFrame(rows, columns=cols).sort_values("event_id").reset_index(drop=True)


def cluster_span_per_event(
    event_ids: pd.Series | np.ndarray, cluster_labels: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Distinct 03b clusters per event: does one fire span several regimes?"""
    ev = pd.Series(np.asarray(event_ids)).reset_index(drop=True)
    cl = pd.Series(np.asarray(cluster_labels)).reset_index(drop=True)
    if len(ev) != len(cl):
        raise ValueError(f"length mismatch: {len(ev)} != {len(cl)}")
    grouped = cl.groupby(ev)
    order = sorted(ev.unique().tolist())
    return pd.DataFrame(
        {
            "event_id": order,
            "n_clusters": [grouped.get_group(e).nunique() for e in order],
            "cluster_ids": [
                ";".join(map(str, sorted(grouped.get_group(e).unique().tolist()))) for e in order
            ],
        }
    )


def anomaly_density_per_event(
    event_ids: pd.Series | np.ndarray, anomaly_flag: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Per-event anomaly load: are multi-day events anomaly-dense or poor?"""
    ev = pd.Series(np.asarray(event_ids)).reset_index(drop=True)
    fl = pd.Series(np.asarray(anomaly_flag)).reset_index(drop=True).fillna(False).astype(bool)
    if len(ev) != len(fl):
        raise ValueError(f"length mismatch: {len(ev)} != {len(fl)}")
    grouped = fl.groupby(ev)
    order = sorted(ev.unique().tolist())
    return pd.DataFrame(
        {
            "event_id": order,
            "n": [len(grouped.get_group(e)) for e in order],
            "n_flagged": [int(grouped.get_group(e).sum()) for e in order],
        }
    ).assign(frac_flagged=lambda d: d["n_flagged"] / d["n"])
