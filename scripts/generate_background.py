"""Generate the synthetic background sample (random Spanish-land points)."""

from __future__ import annotations

import argparse

from wildfire.data.background import DEFAULT_OUTPUT, generate_background


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random background rows with the merged FIRMS schema."
    )
    parser.add_argument("--n", type=int, default=100_000, help="Rows (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--start", default="2023-01-01", help="Window start (default: 2023-01-01)")
    parser.add_argument("--end", default="2024-12-31", help="Window end (default: 2024-12-31)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output path")
    args = parser.parse_args()

    generate_background(
        n=args.n, seed=args.seed, start=args.start, end=args.end, output=args.output
    )


if __name__ == "__main__":
    main()
