"""Presupuestos autonómicos de prevención y extinción (fuego).

Carga ``data/raw/ccaa/Datos_CCAA_Presupuesto - Datos y Variables Model.csv``
(17 CCAA × 2020-2025, importes en M€ con coma decimal española) y calcula,
para cada fila del dataset FIRMS, la suma de ``Prevención_M€`` y
``Extinción_M€`` de los últimos ``n`` años hasta el año de la detección
(``acq_date``), anclada a la comunidad autónoma asignada por geopandas
(columna ``ccaa``).

Notas
-----
- El CSV usa coma decimal (``"133,4"``): se lee con
  ``pd.read_csv(..., decimal=",")`` para que las columnas de importes
  salgan numéricas. Codificación UTF-8 sin BOM.
- Ventana ``[año - n + 1 .. año]``: los años que no existen en el fichero
  (p. ej. 2019 con n=5) no aportan importe; si la ventana no tiene ningún
  año disponible (p. ej. región sin asignar), el resultado es ``NaN``,
  nunca 0 (``sum(min_count=1)``).
- Una fila por (CCAA, año) en el fichero: no hay duplicados que inflen
  las sumas.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from wildfire.config import PROJECT_ROOT, load_config

# Columnas del CSV de presupuestos (nombres originales, con acentos y €).
REGION_COL = "Comunidad Autónoma"
YEAR_COL = "Año"
PREVENTION_COL = "Prevención_M€"
EXTINCTION_COL = "Extinción_M€"

# Columnas de salida: contrato con la celda final del notebook 01.
SUM_PREVENTION_COL = "sum_prevention"
SUM_EXTINCTION_COL = "sum_extinction"


def load_ccaa_budget(path: Path | None = None) -> pd.DataFrame:
    """Carga el CSV de presupuestos por CCAA con importes numéricos.

    Parameters
    ----------
    path:
        Ruta del CSV. Por defecto ``ccaa.path`` de ``configs/project.yaml``
        resuelto contra ``PROJECT_ROOT``.

    Returns
    -------
    pd.DataFrame
        Una fila por (CCAA, año); columnas de importes y ``% Ejecución``
        numéricas (la coma decimal se convierte en punto).
    """
    if path is None:
        path = PROJECT_ROOT / load_config()["ccaa"]["path"]
    return pd.read_csv(path, decimal=",")


def add_ccaa_budget_sums(
    df: pd.DataFrame,
    n: int | None = None,
    budget: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Añade las sumas de prevención/extinción de los últimos ``n`` años.

    Por cada fila toma su año (``acq_date``) y su comunidad (``ccaa``) y
    suma los importes de ``PREVENTION_COL``/``EXTINCTION_COL`` del
    presupuesto en la ventana ``[año - n + 1 .. año]``.

    Parameters
    ----------
    df:
        DataFrame con las columnas ``ccaa`` (nombre de la comunidad, tal
        cual aparece en el CSV de presupuestos) y ``acq_date`` (ISO).
    n:
        Tamaño de la ventana. Por defecto ``ccaa.years_window`` de
        ``configs/project.yaml``.
    budget:
        Presupuestos ya cargados (para tests / reutilización). Por
        defecto se llama a :func:`load_ccaa_budget`.

    Returns
    -------
    pd.DataFrame
        Copia de ``df`` con dos columnas nuevas al final:
        ``sum_prevention`` y ``sum_extinction`` (float64, ``NaN`` cuando
        la ventana no aporta ningún año, p. ej. región sin asignar).

    Raises
    ------
    KeyError
        Si faltan las columnas ``ccaa`` o ``acq_date`` (la celda de
        geopandas no se ha ejecutado aún).
    ValueError
        Si ``n < 1``.
    """
    if n is None:
        n = load_config()["ccaa"]["years_window"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be an integer >= 1, got {n!r}")

    missing = [col for col in ("ccaa", "acq_date") if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}: run the region (geopandas) step first"
        )

    if budget is None:
        budget = load_ccaa_budget()

    # Una (región, año) duplicada inflaría la suma al hacer merge: se queda
    # con la primera y avisa en vez de corromper en silencio.
    dupes = int(budget.duplicated(subset=[REGION_COL, YEAR_COL]).sum())
    if dupes:
        warnings.warn(f"{dupes} duplicated (region, year) budget rows; keeping first")
        budget = budget.drop_duplicates(subset=[REGION_COL, YEAR_COL], keep="first")

    out = df.copy()
    row_ids = np.arange(len(out))
    # Fechas malas → NaT → año NaN → la ventana no casa con ningún año del
    # presupuesto → sumas NaN (nunca 0, nunca crash, nunca ventana errónea).
    years = pd.to_datetime(
        out["acq_date"], format="%Y-%m-%d", errors="coerce"
    ).dt.year.to_numpy()

    # Una fila candidata por (fila original, desplazamiento de año):
    # n=3 → offsets [-2, -1, 0], es decir [año-2 .. año].
    offsets = np.arange(1 - n, 1)
    window = pd.DataFrame(
        {
            "row_id": np.repeat(row_ids, n),
            "ccaa": np.repeat(out["ccaa"].to_numpy(), n),
            YEAR_COL: (years[:, None] + offsets).ravel(),
        }
    )

    merged = window.merge(
        budget[[REGION_COL, YEAR_COL, PREVENTION_COL, EXTINCTION_COL]],
        left_on=["ccaa", YEAR_COL],
        right_on=[REGION_COL, YEAR_COL],
        how="left",
    )
    # min_count=1: grupo sin ningún año en ventana → NaN (no 0)
    sums = (
        merged.groupby("row_id")[[PREVENTION_COL, EXTINCTION_COL]]
        .sum(min_count=1)
        .reindex(row_ids)
    )
    out[SUM_PREVENTION_COL] = sums[PREVENTION_COL].to_numpy()
    out[SUM_EXTINCTION_COL] = sums[EXTINCTION_COL].to_numpy()
    return out
