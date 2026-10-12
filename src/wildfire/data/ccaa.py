"""Presupuestos autonómicos de prevención y extinción (fuego).

Carga ``data/raw/ccaa/Datos_CCAA_Presupuesto - Datos y Variables Model.csv``
(17 CCAA × 2020-2025, importes en M€ con coma decimal española) y calcula,
para cada fila del dataset FIRMS, tres columnas ancladas al año de la
detección (``acq_date``) y a la comunidad asignada por geopandas (``ccaa``):

- ``prevention_current``: ``Prevención_M€`` del año de la fila (la
  prevención se presupuesta al inicio del año → conocida en el momento
  del fuego, sin fuga).
- ``prevention_prev{N}``: suma de ``Prevención_M€`` de los N años previos.
- ``extinction_prev{N}``: suma de ``Extinción_M€`` de los N años previos
  (**sin el año en curso**: la extinción se gasta durante el año, así que
  incluirlo filtraría al modelo un importe desconocido en enero/febrero).

N sale de ``ccaa.budget_lookback_years`` (``configs/project.yaml``) y forma
parte del nombre de las columnas (``prev{N}``), de modo que el esquema
refleja el valor usado en cada ejecución.

Notas
-----
- El CSV usa coma decimal (``"133,4"``): se lee con
  ``pd.read_csv(..., decimal=",")`` para que las columnas de importes
  salgan numéricas. Codificación UTF-8 sin BOM.
- Años fuera del rango del fichero (p. ej. 2019) no aportan importe; si
  la ventana no tiene ningún año disponible (p. ej. región sin asignar),
  el resultado es ``NaN``, nunca 0 (``sum(min_count=1)``).
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

# Columna de salida con nombre fijo: prevención del año de la fila.
PREVENTION_CURRENT_COL = "prevention_current"


def prevention_prev_col(lookback_years: int) -> str:
    """Nombre de la columna de prevención previa (``prev{N}``)."""
    return f"prevention_prev{lookback_years}"


def extinction_prev_col(lookback_years: int) -> str:
    """Nombre de la columna de extinción previa (``prev{N}``)."""
    return f"extinction_prev{lookback_years}"


def _resolve_lookback(lookback_years: int | None) -> int:
    """Resuelve el lookback (parámetro o ``ccaa.budget_lookback_years``)."""
    if lookback_years is None:
        lookback_years = load_config()["ccaa"]["budget_lookback_years"]
    if (
        isinstance(lookback_years, bool)
        or not isinstance(lookback_years, int)
        or lookback_years < 1
    ):
        raise ValueError(
            f"lookback_years must be an integer >= 1, got {lookback_years!r}"
        )
    return lookback_years


def budget_sum_columns(lookback_years: int | None = None) -> list[str]:
    """Las tres columnas de presupuesto que produce una ejecución.

    Única fuente de verdad para los nombres (el ``N`` de ``prev{N}`` es el
    lookback efectivo de la ejecución, no un literal).
    """
    n = _resolve_lookback(lookback_years)
    return [
        PREVENTION_CURRENT_COL,
        prevention_prev_col(n),
        extinction_prev_col(n),
    ]


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
    budget: pd.DataFrame | None = None,
    lookback_years: int | None = None,
) -> pd.DataFrame:
    """Añade prevención actual + sumas previas de prevención/extinción.

    Por cada fila toma su año (``acq_date``) y su comunidad (``ccaa``):
    ``prevention_current`` es ``Prevención_M€`` de ese año;
    ``prevention_prev{N}`` / ``extinction_prev{N}`` suman los N años
    previos (el año en curso queda fuera de la extinción para no filtrar
    gasto aún no ejecutado).

    Parameters
    ----------
    df:
        DataFrame con las columnas ``ccaa`` (nombre de la comunidad, tal
        cual aparece en el CSV de presupuestos) y ``acq_date`` (ISO).
    budget:
        Presupuestos ya cargados (para tests / reutilización). Por
        defecto se llama a :func:`load_ccaa_budget`.
    lookback_years:
        Años previos que se suman (N de ``prev{N}``). Por defecto
        ``ccaa.budget_lookback_years`` de ``configs/project.yaml``.

    Returns
    -------
    pd.DataFrame
        Copia de ``df`` con las tres columnas nuevas al final (float64,
        ``NaN`` cuando la ventana no aporta ningún año, p. ej. región sin
        asignar).

    Raises
    ------
    KeyError
        Si faltan las columnas ``ccaa`` o ``acq_date`` (la celda de
        geopandas no se ha ejecutado aún).
    ValueError
        Si ``lookback_years < 1``.
    """
    n = _resolve_lookback(lookback_years)
    prev_prev_col = prevention_prev_col(n)
    prev_ext_col = extinction_prev_col(n)

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
    # presupuesto → NaN en las tres columnas (nunca 0, nunca crash).
    years = pd.to_datetime(
        out["acq_date"], format="%Y-%m-%d", errors="coerce"
    ).dt.year.to_numpy()
    amounts = budget[[REGION_COL, YEAR_COL, PREVENTION_COL, EXTINCTION_COL]]

    # Año en curso: una candidata por fila (offset 0). El merge `left`
    # conserva el orden de la izquierda → el vector resultante alinea 1:1
    # con las filas; sin coincidencia queda NaN.
    current = pd.DataFrame(
        {
            "row_id": row_ids,
            "ccaa": out["ccaa"].to_numpy(),
            YEAR_COL: years,
        }
    ).merge(
        amounts,
        left_on=["ccaa", YEAR_COL],
        right_on=[REGION_COL, YEAR_COL],
        how="left",
    )
    out[PREVENTION_CURRENT_COL] = current[PREVENTION_COL].to_numpy()

    # Años previos: una candidata por (fila, desplazamiento) con offsets
    # [-n .. -1]; el año en curso nunca entra (sin fuga de extinción).
    offsets = np.arange(-n, 0)
    window = pd.DataFrame(
        {
            "row_id": np.repeat(row_ids, n),
            "ccaa": np.repeat(out["ccaa"].to_numpy(), n),
            YEAR_COL: (years[:, None] + offsets).ravel(),
        }
    )
    merged = window.merge(
        amounts,
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
    out[prev_prev_col] = sums[PREVENTION_COL].to_numpy()
    out[prev_ext_col] = sums[EXTINCTION_COL].to_numpy()
    return out
