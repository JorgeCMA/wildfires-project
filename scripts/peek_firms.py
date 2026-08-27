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

# --- CLCPlus lookup for first row ---
row = combined.iloc[0]
lat, lon = row["latitude"], row["longitude"]

# Convert to EPSG:3035
t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
x, y = t.transform(lon, lat)

# Find the tile: divide by 100000 and floor
e3035, n3035 = int(x // 100000), int(y // 100000)
tile_name = f"E{e3035}N{n3035}"
print(f"\n--- CLCPlus lookup for first row ---")
print(f"lat={lat}, lon={lon}")
print(f"EPSG:3035: E={x:.2f}, N={y:.2f}")
print(f"Tile: {tile_name}")

tiles = list_clc_tiles()
matches = [t for t in tiles if tile_name in t.name]
if not matches:
    print(f"No tile found for {tile_name}")
else:
    tile_path = matches[0]
    with rasterio.open(tile_path) as src:
        row_px, col_px = ~src.transform * (x, y)
        r, c = int(row_px), int(col_px)
        val = src.read(1, window=Window(c, r, 1, 1))[0, 0]
        print(f"Tile file: {tile_path.name}")
        print(f"Pixel: row={r}, col={c}")
        print(f"CLC value: {val}")
