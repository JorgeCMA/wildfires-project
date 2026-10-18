"""Deployment-unit helpers (PLAN §Plan D).

The occurrence model produces a **ranking**, not a calibrated probability: the
fire/background prevalence is set by the negative sampling (``background_ratio``),
not observed. The cheap, honest companion reported alongside the scores is the
**empirical CCAA-day fire rate** — ``P(>=1 detection in a CCAA on a day)``,
estimated from the FIRMS detections against the full CCAA x day calendar. It
uses no extra API quota and gives the rare-event base rate a plan can act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

DATE_COL = "acq_date"
CCAA_COL = "ccaa"
ALL_LABEL = "ALL"


def ccaa_day_fire_rates(
    positives: pd.DataFrame,
    *,
    date_col: str = DATE_COL,
    ccaa_col: str = CCAA_COL,
) -> pd.DataFrame:
    """Empirical fire-day rate per CCAA (PLAN §Plan D).

    A *fire day* is a ``(ccaa, date)`` cell with at least one detection. The
    denominator is every day in the observations' calendar span for each CCAA
    that appears, so ``fire_rate`` is ``P(>=1 detection | CCAA, day)``.

    Parameters
    ----------
    positives:
        Detection frame with a parseable ``date_col`` and a ``ccaa_col``.
    date_col, ccaa_col:
        Column names to use (defaults ``acq_date`` / ``ccaa``).

    Returns
    -------
    pd.DataFrame
        One row per CCAA (descending ``fire_rate``) plus an ``ALL`` row with the
        pooled CCAA-day rate, with columns ``ccaa``, ``n_ccaa_days``,
        ``n_fire_days`` and ``fire_rate``.
    """
    missing = [c for c in (date_col, ccaa_col) if c not in positives.columns]
    if missing:
        raise KeyError(f"positives missing columns: {missing}")

    dates = pd.to_datetime(positives[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_col} has unparseable values")

    days = dates.dt.normalize()
    span = pd.date_range(days.min(), days.max(), freq="D")
    n_days = len(span)
    work = positives.assign(_day=days)

    rows = []
    for ccaa, group in work.groupby(ccaa_col):
        n_fire_days = int(group["_day"].nunique())
        rows.append(
            {
                "ccaa": ccaa,
                "n_ccaa_days": n_days,
                "n_fire_days": n_fire_days,
                "fire_rate": n_fire_days / n_days,
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("fire_rate", ascending=False)
        .reset_index(drop=True)
    )

    n_ccaa = int(work[ccaa_col].nunique())
    total_cells = n_days * n_ccaa
    total_fire_days = int(table["n_fire_days"].sum())
    all_row = pd.DataFrame(
        [
            {
                "ccaa": ALL_LABEL,
                "n_ccaa_days": total_cells,
                "n_fire_days": total_fire_days,
                "fire_rate": total_fire_days / total_cells,
            }
        ]
    )
    return pd.concat([table, all_row], ignore_index=True)


#: Ordered triage labels by number of bands (PLAN §Plan D deployment unit).
RISK_BAND_LABELS: dict[int, list[str]] = {
    3: ["Bajo", "Medio", "Alto"],
    5: ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"],
}


def _resolve_band_labels(n_bands: int, labels: list[str] | None) -> list[str]:
    if labels is not None:
        if len(labels) != n_bands:
            raise ValueError(f"labels has {len(labels)} entries, expected {n_bands}")
        return list(labels)
    if n_bands not in RISK_BAND_LABELS:
        raise ValueError(
            f"no default labels for n_bands={n_bands}; pass labels explicitly"
        )
    return list(RISK_BAND_LABELS[n_bands])


def quantile_bands(
    scores: pd.Series,
    *,
    n_bands: int = 5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Relative triage bands + percentile for a 1-D score series.

    Bands are **relative to the scored batch**: a top-band row is high compared
    with the other rows scored together, not against an absolute danger scale
    (the occurrence score is a ranking, PLAN §Plan D). Percentiles are
    rank-based, so identical scores always land in the same band.

    Returns a frame aligned to ``scores`` with ``risk_band`` (ordered
    categorical) and ``risk_percentile`` (0-100).
    """
    s = pd.Series(scores, dtype="float64")
    if s.isna().any():
        raise ValueError("quantile_bands needs finite scores (found NaN)")
    resolved = _resolve_band_labels(n_bands, labels)
    percentile = s.rank(method="average", pct=True) * 100.0
    edges = np.linspace(0.0, 100.0, n_bands + 1)
    band = pd.cut(percentile, bins=edges, labels=resolved, include_lowest=True)
    return pd.DataFrame(
        {"risk_band": band, "risk_percentile": percentile}, index=s.index
    )


def add_risk_bands(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    group_col: str | None = "protocol",
    n_bands: int = 5,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Append ``risk_band``/``risk_percentile``, banding within ``group_col``.

    Bands are computed separately per group (e.g. per CV protocol) because
    scores coming from different folds are not on a common scale. With
    ``group_col=None`` the whole frame is banded together.
    """
    if score_col not in frame.columns:
        raise KeyError(f"frame lacks score column {score_col!r}")
    if group_col is not None and group_col not in frame.columns:
        raise KeyError(f"frame lacks group column {group_col!r}")
    resolved = _resolve_band_labels(n_bands, labels)

    bands = pd.Series(pd.NA, index=frame.index, dtype="object")
    percentiles = pd.Series(np.nan, index=frame.index, dtype="float64")
    groups = (
        frame.groupby(group_col, sort=False)
        if group_col is not None
        else [(None, frame)]
    )
    for _, chunk in groups:
        band_frame = quantile_bands(chunk[score_col], n_bands=n_bands, labels=resolved)
        bands.loc[chunk.index] = band_frame["risk_band"].to_numpy()
        percentiles.loc[chunk.index] = band_frame["risk_percentile"].to_numpy()

    out = frame.copy()
    out["risk_band"] = pd.Categorical(bands, categories=resolved, ordered=True)
    out["risk_percentile"] = percentiles
    return out


def join_ccaa_fire_rates(
    records: pd.DataFrame,
    ccaa_of_record: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    record_col: str = "record_id",
    ccaa_col: str = CCAA_COL,
    rate_col: str = "fire_rate",
    out_col: str = "ccaa_fire_rate",
) -> pd.DataFrame:
    """Attach the empirical CCAA-day fire rate to scored rows.

    ``ccaa_of_record`` maps ``record_col`` -> CCAA (deduplicated); the pooled
    ``ALL`` row of ``rates`` is excluded, so ``out_col`` is the per-region base
    rate (NaN where a region is unknown). The ranking score is unchanged.
    """
    for name, frame, col in (
        ("records", records, record_col),
        ("ccaa_of_record", ccaa_of_record, record_col),
        ("rates", rates, ccaa_col),
        ("rates", rates, rate_col),
    ):
        if col not in frame.columns:
            raise KeyError(f"{name} lacks column {col!r}")
    mapping = ccaa_of_record[[record_col, ccaa_col]].drop_duplicates(
        subset=[record_col]
    )
    region_rates = rates.loc[rates[ccaa_col] != ALL_LABEL, [ccaa_col, rate_col]]
    out = records.merge(mapping, on=record_col, how="left", validate="many_to_one")
    out = out.merge(region_rates, on=ccaa_col, how="left", validate="many_to_one")
    return out.rename(columns={rate_col: out_col})


__all__ = [
    "ALL_LABEL",
    "CCAA_COL",
    "DATE_COL",
    "RISK_BAND_LABELS",
    "add_risk_bands",
    "ccaa_day_fire_rates",
    "join_ccaa_fire_rates",
    "quantile_bands",
]
