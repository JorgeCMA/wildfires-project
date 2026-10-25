"""Read-only status of the supervised background pool.

Thin CLI over :func:`wildfire.analysis.status.pool_status`: reports how far the
background-weather fill has advanced and whether the pool reaches
``background_ratio`` negatives per positive per year. Never spends Open-Meteo
quota nor writes a CSV.

    python scripts/background_pool_status.py
"""

from __future__ import annotations

from wildfire.analysis.status import format_pool_status, pool_status


def main() -> None:
    print(format_pool_status(pool_status()))


if __name__ == "__main__":
    main()
