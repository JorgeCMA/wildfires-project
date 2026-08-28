"""Quick look at all FIRMS CSVs with unified confidence mapping + CLC enrichment."""

from pathlib import Path

import pandas as pd

from wildfire.data.firms import load_all_firms
from wildfire.enrichment.clc_enrichment import load_clc_classes, enrich_with_clc
from wildfire.processing.confidence import add_unified_confidence

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FIRMS ---
combined = add_unified_confidence(load_all_firms(country="Spain", years=[2023, 2024]))

print(f"Combined: {len(combined):,} rows x {combined.shape[1]} columns")
print(f"Years: {sorted(combined['year'].unique())}")
print(f"Sensors: {combined['sensor'].unique().tolist()}")
print(f"\nConfidence distribution:")
print(combined["confidence_cat"].value_counts().to_string())

# --- CLC enrichment ---
print(f"\nRunning CLC enrichment...")
enriched = enrich_with_clc(combined)

cols = ["latitude", "longitude", "acq_date", "sensor", "clc_class", "clc_name"]
print(enriched[cols].head(10).to_string())

# --- Save FIRMS + CLC dataset ---
firms_path = OUT_DIR / "firms_clc.csv"
enriched.to_csv(firms_path, index=False)
print(f"\nSaved FIRMS+CLC dataset: {firms_path} ({len(enriched):,} rows)")

# --- Save CLC terrain reference table ---
clc_terrain = load_clc_classes()
terrain_path = OUT_DIR / "clc_terrain.csv"
clc_terrain.to_csv(terrain_path, index=False)
print(f"Saved CLC terrain table: {terrain_path} ({len(clc_terrain)} classes)")
print(clc_terrain[["clc_code", "clc_name"]].to_string(index=False))
