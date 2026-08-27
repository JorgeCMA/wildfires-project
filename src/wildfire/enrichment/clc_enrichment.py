"""Enrich FIRMS data with CLCPlus Backbone land cover classification."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildfire.config import load_config
from wildfire.data.clc import read_pixel_value
from wildfire.geo.tiles import find_tile_and_pixel

# CLC class label mapping (subset relevant to Spain).
# Full legend: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018-clc2021
CLC_LABELS: dict[int, str] = {
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks and associated land",
    123: "Port areas",
    124: "Airports",
    131: "Mineral extraction sites",
    132: "Dump sites",
    133: "Construction sites",
    141: "Green urban areas",
    142: "Sport and leisure facilities",
    211: "Non-irrigated arable land",
    212: "Permanently irrigated land",
    213: "Rice fields",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    223: "Olive groves",
    231: "Pastures",
    241: "Annual crops associated with permanent crops",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture with significant areas of natural vegetation",
    251: "Agro-forestry areas",
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    323: "Sclerophyllous vegetation",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    334: "Burnt areas",
    335: "Glaciers and perpetual snow",
    411: "Inland marshes",
    412: "Peat bogs",
    421: "Salt marshes",
    422: "Salines",
    423: "Intertidal flats",
    511: "Water courses",
    512: "Water bodies",
    513: "Coastal lagoons",
    521: "Coastal lagoons",
    999: "NODATA",
}


def enrich_with_clc(
    df: pd.DataFrame,
    country: str = "Spain",
    validity: str = "2023-2025",
) -> pd.DataFrame:
    """Add CLCPlus land cover class to each FIRMS row.

    For each fire detection, the code finds the CLCPlus tile containing
    the coordinates and reads the land cover class at that pixel.

    Parameters
    ----------
    df:
        FIRMS DataFrame with ``latitude`` and ``longitude`` columns.
    country:
        Country folder name for CLCPlus tiles.
    validity:
        Validity period folder name.

    Returns
    -------
    pd.DataFrame
        DataFrame with added ``clc_class`` and ``clc_label`` columns.
    """
    df = df.copy()
    clc_classes: list[int | None] = []
    clc_labels_list: list[str | None] = []

    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        result = find_tile_and_pixel(lat, lon, country=country, validity=validity)

        if result is None:
            clc_classes.append(None)
            clc_labels_list.append(None)
            continue

        tile_path, r, c = result
        value = read_pixel_value(tile_path, r, c)
        clc_classes.append(value)
        clc_labels_list.append(CLC_LABELS.get(value, f"Unknown ({value})") if value is not None else None)

    df["clc_class"] = clc_classes
    df["clc_label"] = clc_labels_list
    return df


def save_enriched(df: pd.DataFrame, country: str = "Spain", year: int | None = None) -> Path:
    """Save enriched FIRMS data to ``data/processed/enriched/``.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.
    country:
        Country name.
    year:
        If provided, save as ``firms_{country}_{year}_enriched.csv``.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    config = load_config()
    out_dir = Path(config["output"]["enriched"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_enriched.csv"
    else:
        filename = f"firms_{country.lower()}_enriched.csv"

    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


def load_enriched(country: str = "Spain", year: int | None = None) -> pd.DataFrame:
    """Load previously enriched FIRMS data.

    Parameters
    ----------
    country:
        Country name.
    year:
        If provided, loads ``firms_{country}_{year}_enriched.csv``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the enriched CSV file does not exist.
    """
    config = load_config()
    out_dir = Path(config["output"]["enriched"])

    if year is not None:
        filename = f"firms_{country.lower()}_{year}_enriched.csv"
    else:
        filename = f"firms_{country.lower()}_enriched.csv"

    path = out_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Enriched file not found: {path}")

    return pd.read_csv(path)
