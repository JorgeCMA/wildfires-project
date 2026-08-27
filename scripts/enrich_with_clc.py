"""Enrich merged FIRMS data with CLCPlus land cover."""

from __future__ import annotations

import argparse
import sys

from wildfire.enrichment.clc_enrichment import enrich_with_clc, save_enriched
from wildfire.enrichment.merge_sensors import load_merged
from wildfire.processing.validation import validate_enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich FIRMS data with CLCPlus land cover classification."
    )
    parser.add_argument(
        "--country", default="Spain", help="Country name (default: Spain)"
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Year to enrich (default: all)"
    )
    parser.add_argument(
        "--validity", default="2023-2025",
        help="CLCPlus validity period (default: 2023-2025)"
    )
    args = parser.parse_args()

    print(f"Loading merged FIRMS data for {args.country}...")
    try:
        df = load_merged(country=args.country, year=args.year)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run merge_firms.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Enriching {len(df):,} rows with CLCPlus data...")
    df = enrich_with_clc(df, country=args.country, validity=args.validity)

    warnings = validate_enriched(df)
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    path = save_enriched(df, country=args.country, year=args.year)
    print(f"Saved enriched data to {path}")


if __name__ == "__main__":
    main()
