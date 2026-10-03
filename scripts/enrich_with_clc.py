"""Enrich weather-ready FIRMS data with CLCPlus land cover (pipeline stage).

Equivalente en script de la Celda 4 del notebook 01: parte del dataset con
clima (terminado si existe, si no el parcial filtrado) y escribe la etapa
``firms_spain_weather_clc.csv`` que alimenta a la Celda 6 (CCAA).
"""

from __future__ import annotations

import argparse
import sys

from wildfire.enrichment.clc_enrichment import enrich_with_clc, save_enriched
from wildfire.enrichment.weather_batch import load_weather_for_clc
from wildfire.processing.validation import validate_enriched


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich weather-ready FIRMS data with CLCPlus land cover."
    )
    parser.add_argument(
        "--country", default="Spain", help="Country name (default: Spain)"
    )
    parser.add_argument(
        "--validity", default="2023-2025",
        help="CLCPlus validity period (default: 2023-2025)"
    )
    args = parser.parse_args()

    print("Loading weather-ready FIRMS data (completed or partial)...")
    try:
        df = load_weather_for_clc()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Run scripts/enrich_weather_batch.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Enriching {len(df):,} rows with CLCPlus data...")
    df = enrich_with_clc(df, country=args.country, validity=args.validity)

    warnings = validate_enriched(df)
    if warnings:
        print("Validation warnings:")
        for w in warnings:
            print(f"  - {w}")

    path = save_enriched(df, filename="firms_spain_weather_clc.csv")
    print(f"Saved enriched data to {path}")


if __name__ == "__main__":
    main()
