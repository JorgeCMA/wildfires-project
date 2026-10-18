"""Deployment-unit helpers (PLAN §Plan D).

The occurrence model produces a **ranking**, not a calibrated probability: the
fire/background prevalence is set by the negative sampling (``background_ratio``),
not observed. The cheap, honest companion reported alongside the scores is the
**empirical CCAA-day fire rate** — ``P(>=1 detection in a CCAA on a day)``,
estimated from the FIRMS detections against the full CCAA x day calendar. It
uses no extra API quota and gives the rare-event base rate a plan can act on.
"""

from __future__ import annotations

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


__all__ = ["ALL_LABEL", "CCAA_COL", "DATE_COL", "ccaa_day_fire_rates"]
