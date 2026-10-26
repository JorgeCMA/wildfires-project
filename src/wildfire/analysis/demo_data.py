"""Read-only loaders for the demo app (Streamlit + Folium, PLAN §Plan D).

The demo is a thin view: every function here only *loads and lightens* the
artifacts written by notebooks 01/04 (no plotting, no API quota). ``demo/app.py``
orchestrates. Missing artifacts raise the usual ``FileNotFoundError`` from
``load_csv``; the app catches it and tells the user which notebook to run.

Note: the point map deliberately uses only the **raw detections** (where fires
happened, by date/region); the model's scores live in the Plan A/B/C panels.
"""

from __future__ import annotations

import json

import pandas as pd  # type: ignore[import-untyped]
import shapely  # type: ignore[import-untyped]

from wildfire.analysis.deployment import ALL_LABEL
from wildfire.config import load_config
from wildfire.data.io import load_csv
from wildfire.geo.regions import REGION_NAME_MAP, load_boundaries

DETECTIONS_CSV = "data/processed/expanded/firms_spain_final_lags_index.csv"
OCCURRENCE_SCORES_CSV = "data/processed/predictions/occurrence_scores.csv"
FOLD_METRICS_CSV = "data/processed/predictions/occurrence_fold_metrics.csv"

#: Columns needed for the raw time map (keeps the 23k-row read light).
DETECTION_COLUMNS = ["detection_id", "latitude", "longitude", "acq_date", "ccaa"]

#: Corporate palette (single source of truth for charts/maps). The Streamlit
#: theme in ``.streamlit/config.toml`` mirrors these values.
PALETTE: dict[str, str] = {
    "bg": "#FFF9EA",
    "secondary_bg": "#FFF6E5",
    "text": "#000000",
    "muted": "#5A4A35",
    "primary": "#F65C2C",
    "accent": "#FFC074",
    "gold": "#FDAA15",
    "deep": "#B3261E",
    "link": "#B8401B",
    "grid": "#EADFC8",
}

#: Ordered 5-band palette (matches ``deployment.RISK_BAND_LABELS[5]``), low→high.
RISK_BAND_COLORS: dict[str, str] = {
    "Muy bajo": "#FFF0CF",
    "Bajo": PALETTE["accent"],
    "Medio": PALETTE["gold"],
    "Alto": PALETTE["primary"],
    "Muy alto": PALETTE["deep"],
}


def load_detections(
    *,
    path: str = DETECTIONS_CSV,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Raw FIRMS detections for the time map (lat/lon/date/region)."""
    cols = DETECTION_COLUMNS if columns is None else list(columns)
    frame = load_csv(path, usecols=cols)
    frame["acq_date"] = pd.to_datetime(frame["acq_date"], errors="coerce")
    return frame


def load_scores(
    protocol: str = "season", *, path: str = OCCURRENCE_SCORES_CSV
) -> pd.DataFrame:
    """Out-of-fold scores for one blocked-CV protocol (``season``/``ccaa``)."""
    frame = load_csv(path)
    subset = frame.loc[frame["protocol"] == protocol].copy()
    if subset.empty:
        raise ValueError(f"no rows for protocol {protocol!r} in {path}")
    return subset


def band_counts(scores: pd.DataFrame, *, band_col: str = "risk_band") -> pd.Series:
    """Count of scored rows per risk band, ordered low→high."""
    counts = scores[band_col].value_counts()
    order = [
        band
        for band in ("Muy bajo", "Bajo", "Medio", "Alto", "Muy alto")
        if band in counts.index
    ]
    return counts.reindex(order)


def load_ccaa_rates(*, path: str | None = None) -> pd.DataFrame:
    """Empirical ``P(>=1 detection | CCAA, day)`` table (Plan D unit B)."""
    if path is None:
        path = load_config()["output"]["ccaa_day_fire_rates"]
    return load_csv(path)


def load_reliability(*, path: str | None = None) -> pd.DataFrame:
    """Route-C exceedance-calibration reliability curves, by margin (km)."""
    if path is None:
        path = load_config()["output"]["calibration_reliability"]
    return load_csv(path)


def load_fold_metrics(*, path: str = FOLD_METRICS_CSV) -> pd.DataFrame:
    """Blocked-CV metrics per protocol/fold/model."""
    return load_csv(path)


def _to_multipolygon(geom):
    """Collapse to a Polygon/MultiPolygon (``make_valid`` yields a collection)."""
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return shapely.MultiPolygon([geom])
    if geom.geom_type == "MultiPolygon":
        return geom
    parts: list = []
    for part in shapely.get_parts(geom):
        if part.geom_type == "Polygon":
            parts.append(part)
        elif part.geom_type == "MultiPolygon":
            parts.extend(part.geoms)
    return shapely.MultiPolygon(parts)


def ccaa_base_rate_geojson(
    rates: pd.DataFrame,
    *,
    boundaries=None,
    rate_col: str = "fire_rate",
) -> dict:
    """GeoJSON of the CCAA boundaries with the base rate joined.

    Names are translated to the budget convention (``REGION_NAME_MAP``) so the
    rate table matches; regions without a rate (Ceuta/Melilla, or an unmapped
    GeoJSON name) carry a null ``fire_rate``. The choropleth keys on
    ``feature.properties.ccaa_es``.
    """
    config = load_config()["ccaa"]
    if boundaries is None:
        boundaries = load_boundaries()
    name_field = config["boundaries_name_field"]

    geo = boundaries.copy()
    geo["ccaa_es"] = geo[name_field].map(REGION_NAME_MAP).fillna(geo[name_field])
    region_rates = rates.loc[rates["ccaa"] != ALL_LABEL, ["ccaa", rate_col]]
    merged = geo.merge(region_rates, left_on="ccaa_es", right_on="ccaa", how="left")
    # Keep only JSON-safe properties (the GeoJSON carries a datetime `created_at`)
    # and collapse GeometryCollections so folium can compute map bounds.
    merged[merged.geometry.name] = merged.geometry.apply(_to_multipolygon)
    slim = merged[["ccaa_es", rate_col, merged.geometry.name]]
    return json.loads(slim.to_json())


__all__ = [
    "DETECTIONS_CSV",
    "DETECTION_COLUMNS",
    "FOLD_METRICS_CSV",
    "OCCURRENCE_SCORES_CSV",
    "PALETTE",
    "RISK_BAND_COLORS",
    "band_counts",
    "ccaa_base_rate_geojson",
    "load_ccaa_rates",
    "load_detections",
    "load_fold_metrics",
    "load_reliability",
    "load_scores",
]
