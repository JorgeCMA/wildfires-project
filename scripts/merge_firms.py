"""Merge VIIRS and MODIS FIRMS data into a unified dataset."""

from __future__ import annotations

import argparse
import sys

from wildfire.enrichment.merge_sensors import merge_viirs_modis, save_merged
from wildfire.processing.validation import validate_firms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge VIIRS and MODIS FIRMS data with unified confidence."
    )
    parser.add_argument(
        "--country", default="Spain", help="Country name (default: Spain)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to merge (default: all configured years)"
    )
    args = parser.parse_args()

    print(f"Merging FIRMS data for {args.country}...")
    df = merge_viirs_modis(country=args.country, years=args.years)

    if df.empty:
        print("No data found. Check that raw FIRMS CSVs exist.", file=sys.stderr)
        sys.exit(1)

    warnings = validate_firms(df)
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    path = save_merged(df, country=args.country)
    print(f"Saved {len(df):,} rows to {path}")


if __name__ == "__main__":
    main()
