"""Read-only status of the supervised background pool (demo provisionality).

The occurrence-model negatives live in ``background_days.csv`` and are grown
from the synthetic weather partial (``firms_spain_background_weather_partial.csv``)
by ``notebooks/02d_background_data.ipynb``. Until the pool reaches
``background_ratio`` negatives per positive **per year**, every model-derived
number (PR-AUC/ablation, route-C calibration rates) is provisional.

This script only *reads* and reports; it never spends Open-Meteo quota nor
writes any CSV. Run it to see how far the slow fill has advanced and whether
the pool is ready to freeze::

    python scripts/background_pool_status.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from wildfire.config import PROJECT_ROOT, load_config
from wildfire.data.io import load_csv
from wildfire.enrichment.weather_batch import (
    BACKGROUND_PARTIAL_CSV,
    find_resume_row,
    weather_fields,
)

#: Positives keyed by ``detection_id`` (same file notebook 04 reads).
POSITIVES_CSV = "data/processed/expanded/firms_spain_final_lags_index.csv"


def _year_counts(frame: pd.DataFrame) -> dict[int, int]:
    years = pd.to_datetime(frame["acq_date"], errors="coerce").dt.year
    return {int(y): int(n) for y, n in years.value_counts().items()}


def pool_status() -> dict:
    """Compute the fill + ratio status without writing anything."""
    config = load_config()
    ratio = int(config["modeling"]["occurrence"]["background_ratio"])
    pool_path = config["output"]["background_days"]

    partial = load_csv(BACKGROUND_PARTIAL_CSV)
    fields = weather_fields()
    total = len(partial)
    filled = (
        int(partial["temperature_2m"].notna().sum())
        if "temperature_2m" in partial
        else 0
    )
    complete = int(partial[fields].notna().all(axis=1).sum()) if total else 0
    resume = find_resume_row(partial)

    positives = _year_counts(load_csv(POSITIVES_CSV))
    if (PROJECT_ROOT / Path(pool_path)).exists():
        pool_frame = load_csv(pool_path)
        pool = _year_counts(pool_frame)
        pool_rows = len(pool_frame)
    else:
        pool: dict[int, int] = {}
        pool_rows = 0

    per_year = {}
    ready = True
    for year in sorted(positives):
        needed = ratio * positives[year]
        have = pool.get(year, 0)
        achieved = have / positives[year] if positives[year] else 0.0
        per_year[year] = {
            "positives": positives[year],
            "needed": needed,
            "have": have,
            "achieved": achieved,
        }
        if have < needed:
            ready = False

    return {
        "ratio": ratio,
        "pool_path": str(pool_path),
        "partial_path": str(BACKGROUND_PARTIAL_CSV),
        "partial_total": total,
        "partial_filled": filled,
        "partial_complete": complete,
        "resume_row": resume,
        "pool_rows": pool_rows,
        "per_year": per_year,
        "ready": ready,
    }


def main() -> None:
    status = pool_status()
    total = status["partial_total"]
    pct = 100.0 * status["partial_filled"] / total if total else 0.0

    print("Background pool status (supervised negatives)")
    print("-" * 46)
    print(f"Weather partial : {status['partial_path']}")
    print(f"  rows total    : {total:,}")
    print(f"  with weather  : {status['partial_filled']:,}  ({pct:.1f}%)")
    print(f"  fully complete: {status['partial_complete']:,}")
    print(f"  resume row    : {status['resume_row']:,}")
    print()
    print(f"Enriched pool    : {status['pool_path']}")
    print(f"  rows          : {status['pool_rows']:,}")
    print()
    print(f"Ratio target (background_ratio={status['ratio']}) vs usable pool")
    print(f"  {'year':>6} {'positives':>10} {'needed':>10} {'have':>8} {'ratio':>7}")
    for year, row in status["per_year"].items():
        print(
            f"  {year:>6} {row['positives']:>10,} {row['needed']:>10,} "
            f"{row['have']:>8,} {row['achieved']:>7.2f}"
        )
    print()
    if status["ready"]:
        print("VERDICT: READY - every year reaches the ratio target; safe to freeze.")
    else:
        print(
            "VERDICT: PROVISIONAL - pool short of the ratio target; "
            "fill weather + rebuild 02d (demo must state this)."
        )


if __name__ == "__main__":
    main()
