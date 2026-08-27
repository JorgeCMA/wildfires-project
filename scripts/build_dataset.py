"""Build the final analysis-ready dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from wildfire.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble the final dataset from enriched data."
    )
    parser.add_argument(
        "--country", default="Spain", help="Country name (default: Spain)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="Years to include (default: all available)"
    )
    parser.add_argument(
        "--output", default=None, help="Output CSV path (default: auto-generated)"
    )
    args = parser.parse_args()

    config = load_config()
    enriched_dir = Path(config["output"]["enriched"])

    years = args.years or config["firms"]["years"]
    frames: list[pd.DataFrame] = []

    for year in years:
        path = enriched_dir / f"firms_{args.country.lower()}_{year}_enriched.csv"
        if path.exists():
            df = pd.read_csv(path)
            frames.append(df)
            print(f"Loaded {len(df):,} rows from {path.name}")
        else:
            print(f"Skipping {year}: {path.name} not found")

    if not frames:
        print("No enriched data found. Run enrichment scripts first.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(combined):,} rows across {len(frames)} file(s)")

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(config["output"]["processed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"dataset_{args.country.lower()}_{'_'.join(map(str, years))}.csv"

    combined.to_csv(out_path, index=False)
    print(f"Saved final dataset to {out_path}")


if __name__ == "__main__":
    main()
