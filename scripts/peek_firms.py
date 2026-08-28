"""Quick look at all FIRMS CSVs with unified confidence mapping."""

import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from wildfire.data.clc import list_clc_tiles
from wildfire.data.firms import load_all_firms
from wildfire.processing.confidence import add_unified_confidence

# --- FIRMS ---
combined = add_unified_confidence(load_all_firms(country="Spain", years=[2023, 2024]))

print(f"Combined: {len(combined):,} rows x {combined.shape[1]} columns")
print(f"Years: {sorted(combined['year'].unique())}")
print(f"Sensors: {combined['sensor'].unique().tolist()}")
print(f"\nConfidence distribution:")
print(combined["confidence_cat"].value_counts().to_string())
print()
cols = ["latitude", "longitude", "acq_date", "sensor", "satellite", "frp",
        "confidence_og_cat", "confidence_og_num", "confidence_cat", "confidence_num"]
print(combined[cols].head(10).to_string())