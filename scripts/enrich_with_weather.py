"""Enrich FIRMS data with Open-Meteo weather data."""

from __future__ import annotations

import argparse
import sys

from wildfire.enrichment.weather_enrichment import enrich_with_weather, save_weather_enriched
from wildfire.processing.validation import validate_enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich FIRMS data with Open-Meteo historical weather."
    )
    parser.add_argument(
        "--country", default="Spain", help="Country name (default: Spain)"
    )
    parser.add_argument(
        "--year", type=int, default=None, help="Year to enrich"
    )
    parser.add_argument(
        "--input", required=True, help="Path to the input (enriched) CSV file"
    )
    parser.add_argument(
        "--variables", nargs="+", default=None,
        help="Open-Meteo hourly variables (default: core fire weather)"
    )
    args = parser.parse_args()

    import pandas as pd
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Enriching {len(df):,} rows with weather data...")
    df = enrich_with_weather(df, hourly_variables=args.variables)

    warnings = validate_enriched(df)
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    path = save_weather_enriched(df, country=args.country, year=args.year)
    print(f"Saved weather-enriched data to {path}")


if __name__ == "__main__":
    main()
