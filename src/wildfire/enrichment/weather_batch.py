"""Descarga por lotes de clima Open-Meteo con reintentos según límite de API.

Módulo de producción extraído de ``tests/test_openmeteo_requests.py``: consulta
POST multi-ubicación (``fetch_next_batch``), relleno interpolado de las 15
variables horarias en el CSV parcial, reanudación, y la capa de límites de
tasa que permite ejecutar sin supervisión (``run_weather_enrichment``).

El tier gratuito de Open-Meteo limita a 600/5.000/10.000 llamadas ponderadas
por minuto/hora/día y lo comunica como HTTP 429 con el cuerpo
``{'error': True, 'reason': 'Minutely API request limit exceeded...'}``.
``retry-requests`` no reintenta 429 (solo 500/502/504) y ``requests-cache``
solo cachea 200, así que este módulo es dueño de toda la lógica de reintento:
clasifica el motivo, calcula la espera hasta el reinicio del contador y
reintenta el mismo lote (la reanudación por CSV lo hace a prueba de fallos).
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from openmeteo_requests.Client import OpenMeteoRequestsError

from wildfire.config import PROJECT_ROOT, load_config
from wildfire.enrichment.merge_sensors import load_merged

# Claves cuyo valor varía por fila: se convierten en listas (una entrada por
# ubicación) para la consulta por lotes. El resto (hourly, models, timezone)
# vive en configs/project.yaml bajo ``openmeteo.batch_query``.
ROW_KEYS = ("latitude", "longitude", "start_date", "end_date")

# Dónde se guarda la respuesta HTTP para reutilizarla después
OUTPUT_JSON = (
    PROJECT_ROOT / load_config()["output"]["processed"] / "openmeteo_response.json"
)

# CSV de enriquecimiento parcial: es a la vez entrada (reanudación) y salida.
# Base = merged (sin CLC): el orden del pipeline es merge -> confianza ->
# open-meteo -> CLC, así la enriquecidora de CLC se puede repetir sin volver
# a gastar la API meteorológica. Empieza como copia del merged y va ganando
# las 15 columnas de clima lote a lote (una fila con `temperature_2m` = lote
# ya descargado). Nombre autoexplicativo: denota que contiene clima parcial.
PARTIAL_CSV = (
    PROJECT_ROOT
    / load_config()["output"]["enriched"]
    / "firms_spain_weather_partial.csv"
)

# CSV de clima terminado: lo escribe `finalize_weather_csv` cuando la
# reanudación llega al final (todas las filas con `temperature_2m`). Es la
# entrada de la celda de clima del notebook 01; si no existe, esa celda
# cae en el parcial y se queda con las filas ya rellenadas.
WEATHER_COMPLETE_CSV = (
    PROJECT_ROOT / load_config()["output"]["enriched"] / "firms_spain_weather.csv"
)


def weather_fields() -> list[str]:
    """Las 15 variables horarias pedidas a Open-Meteo (orden de ``hourly``)."""
    return list(load_config()["openmeteo"]["batch_query"]["hourly"])


def build_row_requests(df: pd.DataFrame) -> list[dict]:
    """Un objeto de petición por fila: su lat/lon y su fecha como inicio y fin.

    Réplica de la consulta por lotes a escala de fila: cada detección genera
    su propia petición con ``start_date == end_date == acq_date``, junto con
    los parámetros fijos de ``configs/project.yaml`` (``hourly``, ``models``,
    ``timezone``).
    """
    config = load_config()["openmeteo"]
    # Copia profunda: sin ella los 500 dicts compartirían la MISMA lista
    # `hourly` con la config global y cualquier mutación la corrompería.
    query = copy.deepcopy(config["batch_query"])
    row_requests: list[dict] = []
    for _, row in df.iterrows():
        date = str(row["acq_date"])[:10]
        row_requests.append(
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "start_date": date,
                "end_date": date,
                **query,
            }
        )
    return row_requests


def load_partial_or_merged(path: Path = PARTIAL_CSV) -> pd.DataFrame:
    """CSV de enriquecimiento parcial si existe; si no, el merged completo.

    Parameters
    ----------
    path:
        Ruta del CSV parcial. Por defecto ``PARTIAL_CSV``.
    """
    if path.exists():
        return pd.read_csv(path)
    return load_merged()


def load_weather_for_clc(
    complete_path: Path = WEATHER_COMPLETE_CSV,
    partial_path: Path = PARTIAL_CSV,
) -> pd.DataFrame:
    """Dataset de clima listo para la celda de CLC del notebook 01.

    Flujo (contrato de la celda 3 del notebook):

    1. Si existe el CSV de clima **terminado** → se devuelve tal cual
       (todas las filas ya tienen las 15 variables).
    2. Si no → se parte del **parcial** y solo se quedan las filas con
       valor de ``temperature_2m`` (las descargadas hasta ahora). Así el
       notebook puede avanzar con los ~5.000 primeros llenos mientras la
       descarga sigue por separado.

    Parameters
    ----------
    complete_path:
        CSV terminado. Por defecto ``WEATHER_COMPLETE_CSV``.
    partial_path:
        CSV de reanudación. Por defecto ``PARTIAL_CSV``.

    Returns
    -------
    pd.DataFrame
        Filas con clima, en el mismo orden que el parcial/merged.

    Raises
    ------
    RuntimeError
        Si todavía no hay ningún dato de clima: ni el CSV terminado
        existe ni el parcial tiene la columna ``temperature_2m`` (o la
        tiene pero vacía) → hay que ejecutar el paso de clima primero.
    """
    if complete_path.exists():
        return pd.read_csv(complete_path)

    df = load_partial_or_merged(partial_path)
    if "temperature_2m" not in df.columns:
        raise RuntimeError(
            "No weather data found (no completed CSV and the partial one has "
            "no temperature_2m column): run the Open-Meteo step first "
            "(scripts/enrich_weather_batch.py)"
        )
    filled = df.dropna(subset=["temperature_2m"])
    if filled.empty:
        raise RuntimeError(
            f"No row has a temperature_2m value in {partial_path}: run the "
            "Open-Meteo step first (scripts/enrich_weather_batch.py)"
        )
    # Índice posicional limpio: dropna conserva el índice original y
    # cualquier asignación posicional (iloc) aguas abajo se desalinearía.
    return filled.reset_index(drop=True)


def find_resume_row(df: pd.DataFrame) -> int:
    """Primera fila sin valor de ``temperature_2m`` (dónde sigue la descarga).

    - Sin columna de clima (merged en crudo) → ``0``.
    - Con algunas filas sin rellenar → su posición (p. ej. 500 si 0-499 ya
      tienen valor).
    - Todo rellenado → ``len(df)`` (no queda nada que descargar).
    """
    if "temperature_2m" not in df.columns:
        return 0

    missing = df["temperature_2m"].isna().to_numpy().nonzero()[0]
    return int(missing[0]) if missing.size else len(df)


def incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filas con clima parcial: ``temperature_2m`` presente pero alguna de
    las 15 variables en NaN (p. ej. la API devolvió nulo para esa variable
    en esas fechas, como ``boundary_layer_height`` en filas de 2024).

    Si faltan columnas de clima enteras también cuentan como hueco. Sin
    columna ``temperature_2m`` no hay nada descargado → vacío.
    """
    if "temperature_2m" not in df.columns:
        return df.iloc[0:0]
    fields = weather_fields()
    mask = df["temperature_2m"].notna()
    if not all(f in df.columns for f in fields):
        return df[mask]
    return df[mask & df[fields].isna().any(axis=1)]


def completeness_report(df: pd.DataFrame) -> dict[str, int]:
    """NaN por variable entre las filas con ``temperature_2m``.

    Sirve para decidir si el parcial es publicable
    (:func:`finalize_weather_csv`) y para informar de huecos por variable.
    """
    report: dict[str, int] = {}
    if "temperature_2m" not in df.columns:
        return report
    filled = df["temperature_2m"].notna()
    report["rows_with_weather"] = int(filled.sum())
    for field in weather_fields():
        if field in df.columns:
            report[field] = int(df.loc[filled, field].isna().sum())
        else:
            report[field] = int(filled.sum())
    report["incomplete_rows"] = int(incomplete_rows(df).shape[0])
    return report


def validate_batch_rows(batch: pd.DataFrame) -> list[str]:
    """Errores de datos en un lote, antes de gastar cuota de API.

    Una sola fila con coordenadas no numéricas, fecha ilegible u hora
    inválida hace que la API devuelva 400 para la petición entera de 500
    filas (y el reintento repetiría el mismo lote para siempre). Mejor
    fallar aquí con las filas señaladas que entrar en ese bucle.

    Parameters
    ----------
    batch:
        Sublote de filas a enviar (índice original como referencia).

    Returns
    -------
    list[str]
        Un mensaje por fila problemática (vacío = lote válido).
    """
    errors: list[str] = []
    for idx, row in batch.iterrows():
        for col in ("latitude", "longitude"):
            try:
                value = float(row[col])
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                errors.append(f"row {idx}: {col}={row[col]!r} is not finite")
        try:
            date = pd.to_datetime(row["acq_date"], format="%Y-%m-%d", errors="coerce")
        except (TypeError, ValueError):
            date = pd.NaT
        if pd.isna(date):
            errors.append(
                f"row {idx}: acq_date={row['acq_date']!r} is not ISO YYYY-MM-DD"
            )
        try:
            hour, minutes, _ = split_acq_time(row["acq_time"])
        except (TypeError, ValueError):
            errors.append(f"row {idx}: acq_time={row['acq_time']!r} is not HHMM")
            continue
        if not (0 <= hour <= 23 and 0 <= minutes <= 59):
            errors.append(
                f"row {idx}: acq_time={row['acq_time']!r} out of range (HH 00-23, MM 00-59)"
            )
    return errors


def next_batch(
    df: pd.DataFrame | None = None,
    batch_rows: int | None = None,
) -> pd.DataFrame:
    """Siguiente lote a descargar: ``batch_rows`` filas desde la de reanudación.

    Parameters
    ----------
    df:
        DataFrame de entrada. Por defecto ``load_partial_or_merged()``.
    batch_rows:
        Tamaño del lote. Por defecto ``openmeteo.batch_rows`` de config.
    """
    if df is None:
        df = load_partial_or_merged()
    if batch_rows is None:
        batch_rows = load_config()["openmeteo"]["batch_rows"]

    start = find_resume_row(df)
    return df.iloc[start : start + batch_rows]


def build_batch_request(df: pd.DataFrame) -> tuple[str, dict]:
    """Construye ``(url, params)`` de la consulta por lotes multi-ubicación.

    Latitud, longitud y fechas salen de las filas de entrada: cada una aporta
    su ``latitude``/``longitude`` y su ``acq_date`` como ``start_date`` y
    ``end_date`` (``next_batch`` decide qué filas). El resto de parámetros
    (``hourly``, ``models``, ``timezone``) viene de ``configs/project.yaml``.
    """
    config = load_config()["openmeteo"]
    row_requests = build_row_requests(df)

    # Una lista por clave dinámica: una entrada por ubicación. Open-Meteo
    # admite fechas distintas por ubicación en la misma petición.
    params: dict = {
        key: [row_request[key] for row_request in row_requests] for key in ROW_KEYS
    }
    # Copia profunda por el mismo motivo que en build_row_requests.
    params.update(copy.deepcopy(config["batch_query"]))
    return config["base_url"], params


def _finite_or_none(value: float) -> float | None:
    """Escalar del SDK a ``None`` si es NaN."""
    return None if math.isnan(value) else value


def _response_to_record(response, names: list[str], position: int) -> dict:
    """Convierte una respuesta del SDK en un registro JSON-serializable.

    Función separada (y no código inline en el bucle) a propósito: las
    variables locales de la ventana (``window_start``/``window_end``) no
    deben sombrear jamás el ``start`` posicional del lote en
    :func:`fetch_next_batch` (el shadowing rompió el ``json.dumps`` del
    payload con ``TypeError: Timestamp is not JSON serializable``).

    Parameters
    ----------
    response:
        Una respuesta de ``openmeteo.weather_api`` (objeto SDK).
    names:
        Variables pedidas (``params["hourly"]``): el SDK las mapea por
        posición, así que se verifica el conteo.
    position:
        Índice de la respuesta en el lote (para los mensajes de error).

    Returns
    -------
    dict
        Registro con escalares + ``hourly`` (24 entradas con ``date`` ISO
        y una clave por variable; nulos como ``None``).

    Raises
    ------
    ValueError
        Si el offset UTC no es 0, el conteo de variables no coincide o la
        ventana no son 24 valores horarios desde medianoche.
    """
    # Con timezone=GMT el flatbuffer no trae nombre de zona horaria
    tz_name = (response.Timezone() or b"GMT").decode()
    # El epoch del flatbuffer es UTC absoluto y con timezone=GMT el offset
    # es 0, así que la ventana pedida sale directa: 00:00-23:00 de
    # acq_date en UTC (igual que acq_date/acq_time de FIRMS). Error
    # explícito (no assert: en producción debe fallar con contexto).
    if response.UtcOffsetSeconds() != 0:
        raise ValueError(
            f"Batch response {position}: expected UtcOffsetSeconds()==0 "
            f"(timezone=GMT), got {response.UtcOffsetSeconds()}"
        )

    hourly = response.Hourly()
    # El SDK mapea Variables(i) por posición: si la API devuelve menos
    # (o reordena) las columnas se desplazarían en silencio.
    if hourly.VariablesLength() != len(names):
        raise ValueError(
            f"Batch response {position}: API returned "
            f"{hourly.VariablesLength()} variables, requested {len(names)}"
        )
    interval_s = hourly.Interval()
    window_start = pd.to_datetime(hourly.Time(), unit="s")
    window_end = pd.to_datetime(hourly.TimeEnd(), unit="s")
    # La interpolación asume 24 valores horarios empezando a medianoche.
    if (
        interval_s != 3600
        or window_start.hour != 0
        or (window_end - window_start) != pd.Timedelta(hours=24)
    ):
        raise ValueError(
            f"Batch response {position}: unexpected hourly window "
            f"(start={window_start}, end={window_end}, interval={interval_s}s)"
        )
    hourly_dataframe = pd.DataFrame(
        {
            "date": pd.date_range(
                start=window_start,
                end=window_end,
                freq=pd.Timedelta(seconds=interval_s),
                inclusive="left",
            )
        }
    )
    # El orden de variables es el mismo que se pidió en batch_query
    for index, name in enumerate(names):
        hourly_dataframe[name] = hourly.Variables(index).ValuesAsNumpy()

    serializable = hourly_dataframe.copy()
    serializable["date"] = serializable["date"].map(lambda d: d.isoformat())
    return {
        # NaN → None en escalares (el JSON de evidencia lleva
        # allow_nan=False como canario).
        "latitude": _finite_or_none(response.Latitude()),
        "longitude": _finite_or_none(response.Longitude()),
        "elevation": _finite_or_none(response.Elevation()),
        "timezone": tz_name,
        "utc_offset_seconds": response.UtcOffsetSeconds(),
        "hourly": json.loads(serializable.to_json(orient="records")),
    }


def split_acq_time(acq_time: str | float) -> tuple[int, int, float]:
    """Divide la hora de adquisición FIRMS (``HHMM``) en sus tres partes.

    Parameters
    ----------
    acq_time:
        Hora de la detección en formato FIRMS: ``1345`` = 13:45. Acepta
        ``int``, ``str`` o ``float`` (el CSV puede traerlo como ``221.0``).

    Returns
    -------
    tuple[int, int, float]
        ``(hora, minutos, porcentaje)`` donde el porcentaje es
        ``minutos / 60``: el peso de la hora siguiente en la interpolación.
        Para 1345 → ``(13, 45, 0.75)`` (75% hacia 14:00, 25% hacia 13:00).
    """
    value = str(int(acq_time)).zfill(4)
    hour, minutes = int(value[:2]), int(value[2:])
    return hour, minutes, minutes / 60


def interpolate_hourly_fields(hourly: list[dict], acq_time: str | float) -> dict:
    """Interpola cada variable horaria a la hora exacta de ``acq_time``.

    La ventana que devuelve Open-Meteo empieza a las 00:00 de ``acq_date`` y
    el índice del ``hourly`` coincide con la hora (``[0]`` = 00:00, ``[1]`` =
    01:00 …, ver docs de ``hourly.time``). La detección cae entre ``[hora]``
    y ``[hora + 1]``: el porcentaje de ``split_acq_time`` es el peso de la
    hora siguiente (1345 → 25% de 13:00 + 75% de 14:00).

    Parameters
    ----------
    hourly:
        Lista de registros horarios de la respuesta (24 entradas con ``date``
        + una clave por variable pedida).
    acq_time:
        Hora de la detección (``HHMM``), siempre dentro de ``hourly[0]["date"]``.

    Returns
    -------
    dict
        Un valor interpolado por variable (sin ``date``). Si alguno de los
        dos extremos es ``None`` (``NaN`` serializado a JSON por la API),
        el resultado es ``None``: la fila queda sin dato en el CSV.

    Notes
    -----
    - ``acq_time`` con minutos a 0 devuelve el valor exacto de esa hora.
    - Si la hora es la última de la ventana (23) se usa ese valor: no hay
      hora siguiente dentro del día pedido, así que es el más cercano.
    """
    hour, _minutes, pct = split_acq_time(acq_time)

    base_hour = int(hourly[0]["date"][11:13])
    prev_idx = min(max(hour - base_hour, 0), len(hourly) - 1)
    next_idx = min(prev_idx + 1, len(hourly) - 1)

    interpolated: dict = {}
    for key, prev_value in hourly[prev_idx].items():
        if key == "date":
            continue
        next_value = hourly[next_idx][key]
        # pd.isna cubre None y NaN float (la API puede devolver nulos que
        # no pasaron por None al parsear); con hueco el resultado es None.
        if pd.isna(prev_value) or pd.isna(next_value):
            interpolated[key] = None
            continue
        interpolated[key] = prev_value + (next_value - prev_value) * pct
    return interpolated


def fill_weather(df: pd.DataFrame, start: int, records: list[dict]) -> None:
    """Escribe en ``df`` las variables interpoladas de ``records`` (in place).

    Una fila del CSV por cada registro de la respuesta: la fila ``start + i``
    recibe los valores de ``records[i]`` interpolados a su ``acq_time`` con
    ``interpolate_hourly_fields``. Las columnas de clima se crean a la
    primera (float, ``NaN``); fuera de ``[start, start + len(records))`` no
    se toca nada, así que ``find_resume_row`` avanza solo.

    Parameters
    ----------
    df:
        DataFrame completo (parcial o merged); se modifica in place.
    start:
        Posición de la primera fila del lote (``find_resume_row``).
    records:
        Registros de la respuesta HTTP (uno por fila, en el mismo orden).

    Raises
    ------
    ValueError
        Si ``records`` no tiene exactamente un registro por fila del lote.
    """
    fields = weather_fields()
    for field in fields:
        if field not in df.columns:
            df[field] = float("nan")

    batch = df.iloc[start : start + len(records)]
    # zip strict: una respuesta por fila; si no coincide, corta antes de escribir
    matrix: list[list] = []
    for (_, row), record in zip(batch.iterrows(), records, strict=True):
        row_values = interpolate_hourly_fields(record["hourly"], row["acq_time"])
        # pandas rechaza None en columnas float64: el hueco va como NaN
        matrix.append(
            [
                float("nan") if pd.isna(row_values[field]) else row_values[field]
                for field in fields
            ]
        )

    col_positions = [df.columns.get_loc(field) for field in fields]
    df.iloc[start : start + len(records), col_positions] = matrix


def fetch_next_batch(
    save_path: Path | None = OUTPUT_JSON,
    csv_path: Path | None = PARTIAL_CSV,
) -> None:
    """Descarga el siguiente lote, rellena el CSV parcial y guarda la respuesta.

    Flujo de reanudación: carga el CSV de enriquecimiento parcial (si no
    existe, el merged), busca la primera fila sin ``temperature_2m`` y pide
    ``openmeteo.batch_rows`` filas desde esa posición. Latitud, longitud y
    ``acq_date`` (como fecha inicial y final) salen de cada fila.

    La petición se envía en POST: con 500 filas la URL de un GET supera los
    38 KB y nginx responde ``414 Request-URI Too Large``. Después se
    interpolan las 15 variables horarias a la hora exacta de cada detección
    (``fill_weather``) y se reescribe el CSV completo, así cada lote avanza
    la reanudación.

    Parameters
    ----------
    save_path:
        Si no es ``None``, guarda la respuesta HTTP como JSON (petición +
        un registro por ubicación) en esta ruta, sobrescribiéndola.
    csv_path:
        Si no es ``None``, CSV de reanudación: se lee al empezar y se
        reescribe (merged + las filas ya rellenadas) tras cada lote.

    Raises
    ------
    OpenMeteoRequestsError
        Si la API responde 400/429 u otro fallo HTTP; el mensaje de 429
        contiene el motivo (``Minutely``/``Hourly``/``Daily``...) que
        ``classify_rate_limit`` usa para decidir la espera.
    ValueError
        Si alguna fila del lote trae coordenadas, fecha u hora inválidas
        (:func:`validate_batch_rows`): se falla antes de gastar cuota.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry  # type: ignore[import-untyped]

    df = load_partial_or_merged(csv_path) if csv_path is not None else load_merged()
    start = find_resume_row(df)
    batch = next_batch(df)
    if batch.empty:
        print("Nothing to fetch: every row already has a temperature_2m value")
        return

    # Falla rápido con las filas señaladas: una sola fila envenenada
    # devuelve 400 para el POST entero y el reintento repetiría el lote
    # para siempre. No se gasta cuota hasta que el lote sea válido.
    problems = validate_batch_rows(batch)
    if problems:
        raise ValueError(
            "Refusing to spend API quota on invalid batch rows:\n" + "\n".join(problems)
        )

    # Cliente de la API con caché y reintentos (el código de abajo asume `openmeteo`).
    # allowable_methods incluye POST: la caché clavea por URL+body, así los
    # re-ejecuciones del mismo lote no vuelven a gastar la API (que limita a
    # ~100 peticiones/minuto y responde 429 si se pasa)
    session = requests_cache.CachedSession(
        str(PROJECT_ROOT / ".cache"),
        expire_after=3600,
        allowable_methods=("GET", "POST"),
    )
    session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=session)

    url, params = build_batch_request(batch)

    responses = openmeteo.weather_api(url, params=params, method="POST")

    records: list[dict] = []
    for position, response in enumerate(responses):
        records.append(_response_to_record(response, params["hourly"], position))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            # Posición de la primera fila del lote en el CSV: permite saber
            # a qué filas corresponde esta respuesta aunque avance el resume
            "row_start": start,
            "url": url,
            "params": params,
            "responses": records,
        }
        save_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"Batch of {len(batch)} rows -> {len(records)} responses "
            f"saved to {save_path}"
        )

    # El JSON es la evidencia cruda: si el relleno falla, se re-pide el mismo
    # lote (cached) en el próximo intento sin perder nada
    fill_weather(df, start, records)

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(
            f"Rows {start}-{start + len(records) - 1} filled -> {csv_path} "
            f"(resume now at {find_resume_row(df)})"
        )


def finalize_weather_csv(
    partial_path: Path = PARTIAL_CSV,
    complete_path: Path = WEATHER_COMPLETE_CSV,
    require_complete: bool = True,
) -> Path | None:
    """Copia el parcial completo al CSV de clima terminado (idempotente).

    La llama ``scripts/enrich_weather_batch.py`` cuando
    :func:`run_weather_enrichment` devuelve
    :attr:`WeatherBatchStatus.COMPLETE`. Es una copia, no un renombrado:
    el parcial sigue siendo el punto de reanudación si más adelante se
    vuelven a bajar filas (p. ej. por un re-merge).

    Parameters
    ----------
    partial_path:
        CSV de reanudación. Por defecto ``PARTIAL_CSV``.
    complete_path:
        Destino. Por defecto ``WEATHER_COMPLETE_CSV``.
    require_complete:
        Si es ``True`` (defecto), además de tener ``temperature_2m`` en
        todas las filas exige las 15 variables rellenas
        (:func:`incomplete_rows` vacío): una fila con clima parcial no se
        publica como terminada. Con ``False`` basta el centinela clásico.

    Returns
    -------
    Path | None
        ``complete_path`` si el CSV terminado ya existía o se acaba de
        crear; ``None`` si el parcial todavía está incompleto (no se
        toca nada).

    Raises
    ------
    FileNotFoundError
        Si no existe el parcial.
    """
    if not partial_path.exists():
        raise FileNotFoundError(f"Partial CSV not found: {partial_path}")

    df = load_partial_or_merged(partial_path)
    if find_resume_row(df) < len(df):
        return None
    if require_complete:
        pending = incomplete_rows(df)
        if not pending.empty:
            report = completeness_report(df)
            details = ", ".join(
                f"{field}={n}"
                for field, n in report.items()
                if field not in ("rows_with_weather", "incomplete_rows") and n
            )
            print(
                f"Partial has temperature_2m everywhere but "
                f"{len(pending)} rows miss variables ({details}): "
                f"not publishing {complete_path}"
            )
            return None
    if complete_path.exists():
        return complete_path

    complete_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(complete_path, index=False)
    return complete_path


# ---------------------------------------------------------------------------
# Límites de tasa de Open-Meteo (HTTP 429) — clasificación y esperas
# ---------------------------------------------------------------------------


class RateLimit(Enum):
    """Tipo de límite de tasa devuelto por Open-Meteo (HTTP 429)."""

    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    CONCURRENT = "concurrent"
    UNKNOWN_429 = "unknown_429"


class WeatherBatchStatus(Enum):
    """Resultado de una ejecución de ``run_weather_enrichment``."""

    COMPLETE = "complete"
    DAILY_STOP = "daily_stop"
    MAX_BATCHES = "max_batches"
    WAIT_CAP = "wait_cap"


# Motivos reales del servidor (Sources/App/Helper/RateLimiter.swift), en el
# orden en que aparecen en el cuerpo del 429.
_RATE_LIMIT_MARKERS: tuple[tuple[str, RateLimit], ...] = (
    ("Minutely API request limit", RateLimit.MINUTELY),
    ("Hourly API request limit", RateLimit.HOURLY),
    ("Daily API request limit", RateLimit.DAILY),
    ("Too many concurrent requests", RateLimit.CONCURRENT),
)

# Motivos de límite que no reconocemos (p. ej. un límite mensual futuro):
# tratados como límite "para hoy", no como error de petición.
_GENERIC_LIMIT_MARKERS = ("api request limit", "rate limit")


def classify_reason(reason: str) -> RateLimit | None:
    """Clasifica el texto de un error de la API en un :class:`RateLimit`.

    Parameters
    ----------
    reason:
        Cualquier texto que contenga el motivo: el cuerpo del 429
        (``{'error': True, 'reason': 'Minutely API request limit exceeded...'}``)
        o el mensaje envuelto por ``weather_api``
        (``failed to request ...: {...}``).

    Returns
    -------
    RateLimit | None
        El límite detectado, o ``None`` si no es un límite de tasa (p. ej.
        un 400 con ``Invalid Date``): el llamante debe re-lanzar el error.
    """
    for marker, kind in _RATE_LIMIT_MARKERS:
        if marker in reason:
            return kind
    lowered = reason.lower()
    if any(marker in lowered for marker in _GENERIC_LIMIT_MARKERS):
        return RateLimit.UNKNOWN_429
    return None


def classify_rate_limit(exc: BaseException) -> RateLimit | None:
    """Clasifica una excepción de la API (``OpenMeteoRequestsError``, etc.).

    Ver :func:`classify_reason`; la clasificación es por texto, así que un
    error de HTTP 400 (parámetros inválidos) devuelve ``None`` y se re-lanza.
    """
    return classify_reason(str(exc))


def seconds_until_reset(kind: RateLimit, now: datetime | None = None) -> float | None:
    """Segundos hasta que Open-Meteo reinicie el contador de ``kind``.

    Los contadores del servidor se limpian en la frontera UTC correspondiente
    (minuto, hora y medianoche; ver ``RateLimiter.minutelyCallback``), así
    que la espera es hasta la siguiente frontera + el margen de config.

    Parameters
    ----------
    kind:
        Límite alcanzado (:func:`classify_rate_limit`).
    now:
        Instante actual (UTC). Por defecto ``datetime.now(timezone.utc)``;
        se inyecta para poder testear sin dormir.

    Returns
    -------
    float | None
        Segundos a dormir, o ``None`` si hay que parar en vez de esperar
        (``DAILY``: esperaría hasta medianoche; ``UNKNOWN_429``: no sabemos
        qué contador mirar).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    config = load_config()["openmeteo"]

    if kind is RateLimit.MINUTELY:
        boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        wait = (boundary - now).total_seconds()
        return wait + config["minutely_buffer_seconds"]
    if kind is RateLimit.HOURLY:
        boundary = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        wait = (boundary - now).total_seconds()
        return wait + config["hourly_buffer_seconds"]
    if kind is RateLimit.CONCURRENT:
        return float(config["concurrent_backoff_seconds"])
    return None


def estimate_weight_range(successes: int, limit: int) -> tuple[float, float] | None:
    """Rango ``[bajo, alto)`` del peso por petición a partir de ``successes``.

    Si N peticiones idénticas seguidas en la misma ventana se aceptan y la
    N+1 recibe 429, el limitador (``check`` antes de ``increment``) implica
    ``N * peso >= límite`` y ``(N - 1) * peso < límite``.

    Parameters
    ----------
    successes:
        Peticiones aceptadas antes del 429 (``>= 1``).
    limit:
        Límite de la ventana (600 minuto, 5.000 hora, 10.000 día).

    Returns
    -------
    tuple[float, float] | None
        ``(bajo, alto)`` abierto a la derecha; con ``successes == 1`` el
        alto es ``inf``. ``None`` si ``successes < 1``.
    """
    if successes < 1:
        return None
    low = limit / successes
    high = limit / (successes - 1) if successes > 1 else float("inf")
    return low, high


def intersect_weight_ranges(
    ranges: list[tuple[float, float]],
) -> tuple[float, float] | None:
    """Intersección de rangos de peso (ver :func:`estimate_weight_range`).

    Asume linealidad: un rango medido con un cuerpo de S filas se escala a
    ``batch_rows`` antes de interpolar. Devuelve ``None`` si no se solapan.
    """
    if not ranges:
        return None
    low = max(r[0] for r in ranges)
    high = min(r[1] for r in ranges)
    if low >= high:
        return None
    return low, high


def pacing_advice(weight_per_batch: float) -> dict:
    """Llamadas previstas y ``batch_interval_seconds`` para un peso dado.

    Parameters
    ----------
    weight_per_batch:
        Peso estimado de un lote completo (``openmeteo.batch_rows`` filas).

    Returns
    -------
    dict
        ``per_minute``/``per_hour``/``per_day`` (llamadas del lote que caben
        en cada ventana, ``ceil(límite / peso)`` con mínimo 1) y
        ``interval_seconds``: intervalo proactivo que respeta a la vez el
        límite minuto y el promedio horario (el diario se gestiona parando
        y reanudando, no espaciando).
    """
    per_minute = max(1, math.ceil(600 / weight_per_batch))
    per_hour = max(1, math.ceil(5000 / weight_per_batch))
    per_day = max(1, math.ceil(10000 / weight_per_batch))
    interval = max(60 / per_minute, 3600 / per_hour)
    return {
        "per_minute": per_minute,
        "per_hour": per_hour,
        "per_day": per_day,
        "interval_seconds": interval,
    }


def _log(message: str) -> None:
    """Mensaje con marca de tiempo local para la ejecución sin supervisión."""
    now = datetime.now(timezone.utc).astimezone()
    print(f"[{now:%H:%M:%S}] {message}")


def run_weather_enrichment(
    *,
    max_batches: int | None = None,
    interval_seconds: float | None = None,
    csv_path: Path = PARTIAL_CSV,
    save_path: Path = OUTPUT_JSON,
    fetch: Callable[..., None] = fetch_next_batch,
    sleep: Callable[[float], None] = time.sleep,
    max_consecutive_waits: int | None = None,
) -> WeatherBatchStatus:
    """Descarga todos los lotes pendientes reintentando según el límite.

    Bucle de ejecución sin supervisión sobre :func:`fetch_next_batch`:

    1. Lee el CSV de reanudación; si no queda ninguna fila sin
       ``temperature_2m`` → :attr:`WeatherBatchStatus.COMPLETE`.
    2. Respeta ``max_batches`` (:attr:`WeatherBatchStatus.MAX_BATCHES`) y
       duerme ``interval_seconds`` entre lotes correctos (pausa proactiva
       para no barrer el límite minuto).
    3. Si la API devuelve 429, :func:`classify_rate_limit` decide la espera
       (:func:`seconds_until_reset`) y **se reintenta el mismo lote**: el
       CSV no se tocó, así que la reanudación sigue en la misma fila.
       - ``DAILY``/``UNKNOWN_429`` → no se espera: parada limpia
         (:attr:`WeatherBatchStatus.DAILY_STOP`); re-ejecutar continúa
         donde estaba (el límite diario se reinicia en medianoche UTC).
       - Tras ``max_consecutive_waits`` esperas seguidas del mismo tipo sin
         ningún lote correcto intercalado → :attr:`WeatherBatchStatus.WAIT_CAP`
         (red de seguridad contra un error mal clasificado).
    4. Un error que **no** es límite de tasa (400, fallo de red agotados los
       reintentos de transporte) se re-lanza: hay que mirarlo.

    Parameters
    ----------
    max_batches:
        Lotes máximos en esta ejecución (``None`` = hasta completar).
    interval_seconds:
        Pausa entre lotes correctos. Por defecto
        ``openmeteo.batch_interval_seconds`` de config (0 = solo reactiva).
    csv_path:
        CSV de reanudación (entrada y salida).
    save_path:
        Dónde volcar la respuesta JSON del último lote (``None`` = no).
    fetch:
        Función a llamar por lote; por defecto :func:`fetch_next_batch`.
        Se inyecta en los tests para simular 429 sin red.
    sleep:
        ``time.sleep``; se inyecta en los tests para no dormir de verdad.
    max_consecutive_waits:
        Esperas seguidas del mismo límite antes de parar. Por defecto
        ``openmeteo.max_consecutive_waits`` de config.

    Returns
    -------
    WeatherBatchStatus
        Estado final (ver arriba); nunca lanza por límite de tasa.
    """
    config = load_config()["openmeteo"]
    if interval_seconds is None:
        interval_seconds = config["batch_interval_seconds"]
    if max_consecutive_waits is None:
        max_consecutive_waits = config["max_consecutive_waits"]

    batches = 0
    waits = 0
    last_kind: RateLimit | None = None
    pace = False

    while True:
        df = load_partial_or_merged(csv_path)
        resume = find_resume_row(df)
        if resume >= len(df):
            return WeatherBatchStatus.COMPLETE
        if max_batches is not None and batches >= max_batches:
            _log(f"max_batches={max_batches} alcanzado (reanudación en {resume})")
            return WeatherBatchStatus.MAX_BATCHES

        # Pausa proactiva tras el lote anterior (no antes del primero ni
        # después de una espera de límite, que ya esperó lo suyo)
        if pace and interval_seconds > 0:
            sleep(interval_seconds)
            pace = False

        try:
            fetch(save_path=save_path, csv_path=csv_path)
        except OpenMeteoRequestsError as exc:
            kind = classify_rate_limit(exc)
            if kind is None:
                raise
            wait = seconds_until_reset(kind)
            if wait is None:
                _log(
                    f"Límite {kind.value} alcanzado ({exc}); parada limpia, "
                    "re-ejecutar para reanudar"
                )
                return WeatherBatchStatus.DAILY_STOP
            waits = waits + 1 if kind is last_kind else 1
            last_kind = kind
            if waits > max_consecutive_waits:
                _log(
                    f"{waits} esperas seguidas del límite {kind.value} sin "
                    f"progreso; parada limpia ({exc})"
                )
                return WeatherBatchStatus.WAIT_CAP
            _log(
                f"Límite {kind.value}: esperar {wait:.0f}s "
                f"({waits}/{max_consecutive_waits}) y reintentar el lote"
            )
            sleep(wait)
            continue

        batches += 1
        pace = True
        waits = 0
        last_kind = None
        _log(f"Lote descargado ({batches} en esta ejecución)")
