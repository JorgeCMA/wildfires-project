"""Asignación de cada detección a una comunidad autónoma (geopandas).

Cruza las coordenadas FIRMS (``latitude``/``longitude``, EPSG:4326) con los
límites autonómicos de ``data/raw/ccaa/spain-communities.geojson`` (fuente
decidida por el equipo, 2026-10-02) y escribe la columna ``ccaa`` con el
nombre EXACTO del CSV de presupuestos (``wildfire.data.ccaa``), de modo que
``add_ccaa_budget_sums`` pueda hacer el merge sin pisar acentos ni alias.

Notas
-----
- El cruce se hace en EPSG:3035 (metro) para poder medir distancias:
  1. ``sjoin(predicate="within")`` — el punto cae dentro de un polígono.
  2. Fallback ``sjoin_nearest`` con ``ccaa.nearest_max_distance_m`` para
     puntos sin polígono (costa y frontera PT-FR; hoy 99/32.500, todos a
     <1,7 km). Más lejos del tope → ``ccaa`` NaN.
- Punto en la frontera compartida puede casar con 2 polígonos: se queda con
  el primero (``~index.duplicated(keep="first")``) — nunca se duplican filas.
- Nombres del GeoJSON que no estén en :data:`REGION_NAME_MAP` pasan sin
  traducir (``Ceuta``/``Melilla``: no existen en el presupuesto → las sumas
  salen NaN solas, sin casos especiales).
- Solo se añade la columna ``ccaa`` (la última); ninguna fila se elimina.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from wildfire.config import PROJECT_ROOT, load_config

# Contrato con la celda 6 del notebook 01 y con add_ccaa_budget_sums.
CCAA_COL = "ccaa"
LAT_COL = "latitude"
LON_COL = "longitude"

# El GeoJSON no trae "crs": por especificación GeoJSON = EPSG:4326.
CRS_WGS84 = "EPSG:4326"
# Proyección métrica europea (la misma que usa clc_enrichment).
CRS_METRIC = "EPSG:3035"

# `name` del GeoJSON → nombre EXACTO del CSV de presupuestos (17 CCAA).
# Ceuta/Melilla se quedan fuera a propósito: no están en el presupuesto.
REGION_NAME_MAP: dict[str, str] = {
    "Andalucia": "Andalucía",
    "Aragon": "Aragón",
    "Asturias": "Asturias",
    "Baleares": "Islas Baleares",
    "Canarias": "Canarias",
    "Cantabria": "Cantabria",
    "Castilla-La Mancha": "Castilla-La Mancha",
    "Castilla-Leon": "Castilla y León",
    "Cataluña": "Cataluña",
    "Extremadura": "Extremadura",
    "Galicia": "Galicia",
    "La Rioja": "La Rioja",
    "Madrid": "Madrid",
    "Murcia": "Región de Murcia",
    "Navarra": "Navarra",
    "Pais Vasco": "País Vasco",
    "Valencia": "Comunidad Valenciana",
}


def load_boundaries(path: Path | None = None) -> gpd.GeoDataFrame:
    """Carga los límites autonómicos del GeoJSON del equipo.

    Parameters
    ----------
    path:
        Ruta del GeoJSON. Por defecto ``ccaa.boundaries_path`` de
        ``configs/project.yaml`` resuelto contra ``PROJECT_ROOT``.

    Returns
    -------
    gpd.GeoDataFrame
        19 polígonos (17 CCAA + Ceuta + Melilla) con geometrías reparadas
        (``make_valid`` — la de Canarias viene inválida) y CRS EPSG:4326.

    Raises
    ------
    FileNotFoundError
        Si el fichero no existe.
    """
    if path is None:
        path = PROJECT_ROOT / Path(load_config()["ccaa"]["boundaries_path"])
    if not path.exists():
        raise FileNotFoundError(f"CCAA boundaries not found: {path}")

    boundaries = gpd.read_file(path)
    if boundaries.crs is None:
        boundaries = boundaries.set_crs(CRS_WGS84)
    boundaries["geometry"] = boundaries.geometry.make_valid()
    return boundaries


def assign_ccaa(
    df: pd.DataFrame,
    boundaries: gpd.GeoDataFrame | None = None,
    name_field: str | None = None,
    max_distance_m: float | None = None,
) -> pd.DataFrame:
    """Añade la columna ``ccaa`` con el nombre de la comunidad autónoma.

    Parameters
    ----------
    df:
        DataFrame FIRMS con columnas ``latitude`` y ``longitude`` (grados).
    boundaries:
        Límites ya cargados (tests / reutilización). Por defecto se llama a
        :func:`load_boundaries`.
    name_field:
        Campo del GeoJSON con el nombre de la región. Por defecto
        ``ccaa.boundaries_name_field`` de la config.
    max_distance_m:
        Tope del respaldo ``sjoin_nearest`` para puntos sin polígono.
        Por defecto ``ccaa.nearest_max_distance_m`` (0 = solo ``within``).

    Returns
    -------
    pd.DataFrame
        Copia de ``df`` con una columna nueva al final, ``ccaa``: el nombre
        del presupuesto vía :data:`REGION_NAME_MAP` (los no traducidos pasan
        sin cambiar; sin región → NaN). Ninguna fila se pierde ni se
        reordena; el índice se conserva.

    Raises
    ------
    KeyError
        Si faltan ``latitude``/``longitude`` en ``df`` o ``name_field``
        no está en los límites.
    """
    missing = [col for col in (LAT_COL, LON_COL) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}: FIRMS coordinates are required")

    config = load_config()["ccaa"]
    if name_field is None:
        name_field = config["boundaries_name_field"]
    if max_distance_m is None:
        max_distance_m = config["nearest_max_distance_m"]
    if boundaries is None:
        boundaries = load_boundaries()
    if name_field not in boundaries.columns:
        raise KeyError(
            f"Field {name_field!r} not in boundaries "
            f"(available: {sorted(boundaries.columns)})"
        )

    out = df.copy()
    out[CCAA_COL] = pd.Series(pd.NA, index=out.index, dtype="object")
    if out.empty:
        return out

    # Puntos con el MISMO índice que out: la asignación final es posicional
    # por índice y sobrevive a índices no estándar ([5, 7], ...).
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(out[LON_COL], out[LAT_COL]),
        index=out.index,
        crs=CRS_WGS84,
    ).to_crs(CRS_METRIC)

    regions = boundaries[[name_field, "geometry"]].to_crs(CRS_METRIC)

    joined = gpd.sjoin(points, regions, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched = joined[name_field].isna()
    if max_distance_m and unmatched.any():
        nearest = gpd.sjoin_nearest(
            points.loc[unmatched],
            regions,
            how="left",
            max_distance=max_distance_m,
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        joined.loc[unmatched, name_field] = nearest[name_field]

    raw_names = joined.reindex(out.index)[name_field]
    translated = raw_names.map(REGION_NAME_MAP)
    # Traducidos → nombre del presupuesto; el resto (Ceuta/Melilla, NaN)
    # pasa sin cambiar.
    out[CCAA_COL] = translated.where(translated.notna(), raw_names)
    return out
