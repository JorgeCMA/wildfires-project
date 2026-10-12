"""Collapse cross-sensor duplicate FIRMS detections.

Rule C': within each spatio-temporal candidate group (same ``acq_date``,
|dt| <= ``t_group_minutes``, haversine <= ``d_km``, transitively linked),
drop a MODIS row only if a VIIRS row *in the same group* satisfies both
|dt| <= ``t_pair_minutes`` and haversine <= ``d_km``.

Same-instrument neighbours are never touched: adjacent VIIRS pixels from a
single overpass are valid fire-front signal, not duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildfire.config import load_config

REQUIRED_COLUMNS = ("latitude", "longitude", "acq_date", "acq_time", "instrument")

KEPT = "kept"
DROPPED_MODIS_OVERLAP = "dropped_modis_overlap"

_EARTH_RADIUS_KM = 6371.0088


def _acq_time_to_minutes(value: str | float) -> int:
    """Normalize a FIRMS ``acq_time`` (``HMM``/``HHMM``, e.g. ``221`` -> 02:21)."""
    text = str(int(value)).zfill(4)
    return int(text[:2]) * 60 + int(text[2:4])


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Vectorized haversine distance in kilometres."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class _UnionFind:
    """Minimal union-find for grouping transitively linked detections."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, node: int) -> int:
        parent = self.parent
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(self, first: int, second: int) -> None:
        first, second = self.find(first), self.find(second)
        if first == second:
            return
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def _candidate_groups(
    day: pd.DataFrame, t_group_minutes: int, d_km: float
) -> list[list[int]]:
    """Cluster one date's rows into candidate groups (positional indices)."""
    order = day["acq_minutes"].argsort(kind="stable").to_numpy()
    lat = day["latitude"].to_numpy()[order]
    lon = day["longitude"].to_numpy()[order]
    minutes = day["acq_minutes"].to_numpy()[order]
    size = len(day)
    components = _UnionFind(size)
    for i in range(size):
        for j in range(i + 1, size):
            if minutes[j] - minutes[i] > t_group_minutes:
                break
            if _haversine_km(lat[i], lon[i], lat[j], lon[j]) <= d_km:
                components.union(i, j)
    buckets: dict[int, list[int]] = {}
    for i in range(size):
        buckets.setdefault(components.find(i), []).append(int(order[i]))
    return [members for members in buckets.values() if len(members) > 1]


def collapse_cross_sensor_dupes(
    df: pd.DataFrame,
    t_group_minutes: int = 15,
    t_pair_minutes: int = 10,
    d_km: float = 1.0,
    drop: bool = True,
    modeling: bool = False,
) -> pd.DataFrame:
    """Collapse cross-sensor (MODIS-overlapping-VIIRS) duplicate detections.

    Parameters
    ----------
    df:
        FIRMS detections with at least ``latitude``, ``longitude``,
        ``acq_date``, ``acq_time`` and ``instrument`` (``"MODIS"``/``"VIIRS"``)
        columns. The input is never modified.
    t_group_minutes:
        Time window for building candidate groups (transitive linking).
    t_pair_minutes:
        Tighter pairwise time check a MODIS row must satisfy against a
        same-group VIIRS row to be considered overlapped.
    d_km:
        Haversine distance threshold in kilometres, used both for grouping
        and for the pairwise overlap check.
    drop:
        If True (default), return only kept rows. If False, return all rows
        with their flags so callers can audit what would be removed.
    modeling:
        If True, apply :func:`apply_modeling_schema` before returning
        (drops the ``modeling.drop_columns`` blocklist from config).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with a ``dupe_flag`` column (``"kept"`` or
        ``"dropped_modis_overlap"``). With ``drop=True`` only ``"kept"``
        rows are included. With ``modeling=True`` the schema blocklist
        (including ``dupe_flag``) is also removed.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    minutes = df["acq_time"].map(_acq_time_to_minutes).to_numpy()
    result["acq_minutes"] = minutes
    result["dupe_flag"] = KEPT

    for _, day in result.groupby("acq_date", sort=True):
        positions = day.index.to_numpy()
        day_reset = day.reset_index(drop=True)
        for members in _candidate_groups(day_reset, t_group_minutes, d_km):
            group = day_reset.iloc[members]
            if group["instrument"].nunique() < 2:
                continue
            viirs = group[group["instrument"] == "VIIRS"]
            modis = group[group["instrument"] == "MODIS"]
            if viirs.empty or modis.empty:
                continue
            viirs_lat = viirs["latitude"].to_numpy()
            viirs_lon = viirs["longitude"].to_numpy()
            viirs_min = viirs["acq_minutes"].to_numpy()
            for row in modis.itertuples():
                close_in_time = np.abs(viirs_min - row.acq_minutes) <= t_pair_minutes
                if not close_in_time.any():
                    continue
                distances = _haversine_km(
                    np.full(viirs_lat.shape, row.latitude),
                    np.full(viirs_lon.shape, row.longitude),
                    viirs_lat,
                    viirs_lon,
                )
                if bool(((distances <= d_km) & close_in_time).any()):
                    result.loc[positions[row.Index], "dupe_flag"] = (
                        DROPPED_MODIS_OVERLAP
                    )

    result = result.drop(columns=["acq_minutes"])
    if drop:
        result = result[result["dupe_flag"] == KEPT].reset_index(drop=True)
    if modeling:
        result = apply_modeling_schema(result)
    return result


def apply_modeling_schema(
    df: pd.DataFrame, drop_columns: list[str] | None = None
) -> pd.DataFrame:
    """Drop non-modeling columns listed under ``modeling.drop_columns``.

    Parameters
    ----------
    df:
        Any enrichment-stage DataFrame. The input is never modified.
    drop_columns:
        Explicit blocklist. Defaults to ``modeling.drop_columns`` from
        ``configs/project.yaml``. Names absent from ``df`` are ignored,
        so the same list works before dedup (no ``dupe_flag`` yet) and
        after.

    Returns
    -------
    pd.DataFrame
        A new DataFrame without the listed columns.
    """
    if drop_columns is None:
        config = load_config()
        drop_columns = list(config.get("modeling", {}).get("drop_columns", []))
    present = [col for col in drop_columns if col in df.columns]
    return df.drop(columns=present)
