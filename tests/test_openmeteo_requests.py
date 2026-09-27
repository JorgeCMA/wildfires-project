"""Batch query for the ``openmeteo-requests`` SDK: resumable weather fetch."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT, load_config
from wildfire.enrichment import load_merged

# Claves cuyo valor varía por fila: se convierten en listas (una entrada por
# ubicación) para la consulta por lotes. El resto (hourly, models, timezone)
# vive en configs/project.yaml bajo ``openmeteo.batch_query``.
ROW_KEYS = ("latitude", "longitude", "start_date", "end_date")

# Dónde se guarda la respuesta HTTP para reutilizarla después
OUTPUT_JSON = (
    PROJECT_ROOT / load_config()["output"]["processed"] / "openmeteo_response.json"
)

# CSV de enriquecimiento parcial: es a la vez entrada (reanudación) y salida.
# Todavía no existe: mientras tanto se parte del merged.
PARTIAL_CSV = (
    PROJECT_ROOT
    / load_config()["output"]["enriched"]
    / "firms_spain_enriched_partial.csv"
)


def build_row_requests(df: pd.DataFrame) -> list[dict]:
    """Un objeto de petición por fila: su lat/lon y su fecha como inicio y fin.

    Réplica de la consulta por lotes a escala de fila: cada detección genera
    su propia petición con ``start_date == end_date == acq_date``, junto con
    los parámetros fijos de ``configs/project.yaml`` (``hourly``, ``models``,
    ``timezone``).
    """
    config = load_config()["openmeteo"]
    row_requests: list[dict] = []
    for _, row in df.iterrows():
        date = str(row["acq_date"])[:10]
        row_requests.append(
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "start_date": date,
                "end_date": date,
                **config["batch_query"],
            }
        )
    return row_requests


def load_partial_or_merged() -> pd.DataFrame:
    """CSV de enriquecimiento parcial si existe; si no, el merged completo."""
    if PARTIAL_CSV.exists():
        return pd.read_csv(PARTIAL_CSV)
    return load_merged()


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
    params.update(config["batch_query"])
    return config["base_url"], params


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
        Un valor interpolado por variable (sin ``date``).

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
        interpolated[key] = prev_value + (next_value - prev_value) * pct
    return interpolated


def fetch_next_batch(save_path: Path | None = OUTPUT_JSON) -> None:
    """Descarga el siguiente lote de ``batch_rows`` filas y guarda la respuesta.

    Flujo de reanudación: carga el CSV de enriquecimiento parcial (si no
    existe, el merged), busca la primera fila sin ``temperature_2m`` y pide
    ``openmeteo.batch_rows`` filas desde esa posición. Latitud, longitud y
    ``acq_date`` (como fecha inicial y final) salen de cada fila.

    La petición se envía en POST: con 500 filas la URL de un GET supera los
    38 KB y nginx responde ``414 Request-URI Too Large``.

    Parameters
    ----------
    save_path:
        Si no es ``None``, guarda la respuesta HTTP como JSON (petición +
        un registro por ubicación) en esta ruta, sobrescribiéndola.

    Notes
    -----
    Tras guardar la respuesta se hace ``return``: el relleno del CSV parcial
    con las horas ponderadas de ``interpolate_hourly_fields`` es el siguiente
    paso y todavía no está implementado.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    batch = next_batch()
    if batch.empty:
        print("Nothing to fetch: every row already has a temperature_2m value")
        return

    # Cliente de la API con caché y reintentos (el código de abajo asume `openmeteo`)
    session = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=session)

    url, params = build_batch_request(batch)

    responses = openmeteo.weather_api(url, params=params, method="POST")

    records: list[dict] = []
    for response in responses:
        # Con timezone=GMT el flatbuffer no trae nombre de zona horaria
        timezone = (response.Timezone() or b"GMT").decode()
        # El epoch del flatbuffer es UTC absoluto y con timezone=GMT el offset
        # es 0, así que la ventana pedida sale directa: 00:00-23:00 de
        # acq_date en UTC (igual que acq_date/acq_time de FIRMS).
        assert response.UtcOffsetSeconds() == 0

        hourly = response.Hourly()
        hourly_dataframe = pd.DataFrame(
            {
                "date": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s"),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )
            }
        )
        # El orden de variables es el mismo que se pidió en batch_query
        names = params["hourly"]
        for index, name in enumerate(names):
            hourly_dataframe[name] = hourly.Variables(index).ValuesAsNumpy()

        serializable = hourly_dataframe.copy()
        serializable["date"] = serializable["date"].map(lambda d: d.isoformat())
        records.append(
            {
                "latitude": response.Latitude(),
                "longitude": response.Longitude(),
                "elevation": response.Elevation(),
                "timezone": timezone,
                "utc_offset_seconds": response.UtcOffsetSeconds(),
                "hourly": json.loads(serializable.to_json(orient="records")),
            }
        )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"url": url, "params": params, "responses": records}
        save_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(
            f"Batch of {len(batch)} rows -> {len(records)} responses "
            f"saved to {save_path}"
        )

    return


# ---------------------------------------------------------------------------
# Configuración y reanudación — sin HTTP
# ---------------------------------------------------------------------------


def test_batch_rows_config():
    """``batch_rows`` es un entero positivo y NO se envía a la API."""
    openmeteo = load_config()["openmeteo"]

    assert isinstance(openmeteo["batch_rows"], int)
    assert openmeteo["batch_rows"] > 0
    # Vive fuera de batch_query a propósito: batch_query se copia tal cual
    # a los parámetros HTTP.
    assert "batch_rows" not in openmeteo["batch_query"]


def test_find_resume_row_without_weather_column():
    """Sin columnas de clima (merged en crudo) se empieza por la fila 0."""
    df = pd.DataFrame({"latitude": [40.0, 41.0]})

    assert find_resume_row(df) == 0


def test_find_resume_row_partial_fill():
    """Con filas sin rellenar se continúa por la primera que falta."""
    df = pd.DataFrame({"temperature_2m": [1.0, 2.0, None, None]})

    assert find_resume_row(df) == 2


def test_find_resume_row_all_filled():
    """Todo rellenado -> no queda nada que descargar."""
    df = pd.DataFrame({"temperature_2m": [1.0, 2.0]})

    assert find_resume_row(df) == 2


def test_next_batch_uses_config_batch_rows():
    """El lote sale del merged y mide ``batch_rows`` (hoy resume = 0)."""
    batch_rows = load_config()["openmeteo"]["batch_rows"]
    batch = next_batch(load_merged())

    assert len(batch) == batch_rows
    assert batch.index[0] == 0


def test_next_batch_respects_resume_and_tail():
    """Respeta la fila de reanudación y el lote final puede ser más corto."""
    df = pd.DataFrame(
        {"temperature_2m": [1.0] * 10 + [None] * 90, "latitude": range(100)}
    )

    batch = next_batch(df, batch_rows=5)
    assert list(batch.index) == [10, 11, 12, 13, 14]

    tail = next_batch(df, batch_rows=100)
    assert len(tail) == 90

    full = pd.DataFrame({"temperature_2m": [1.0] * 10})
    assert next_batch(full, batch_rows=5).empty


def test_row_requests_from_first_rows():
    """Las 3 primeras filas del merged generan 3 peticiones con su propia fecha."""
    first = load_merged().head(3)
    row_requests = build_row_requests(first)

    assert len(row_requests) == 3
    for req, (_, row) in zip(row_requests, first.iterrows(), strict=True):
        assert req["latitude"] == row["latitude"]
        assert req["longitude"] == row["longitude"]
        assert req["start_date"] == str(row["acq_date"])[:10]
        assert req["end_date"] == req["start_date"]
        assert req["hourly"] == load_config()["openmeteo"]["batch_query"]["hourly"]


def test_build_batch_request_from_first_rows():
    """La consulta por lotes toma lat/lon/fechas de las filas dadas."""
    first = load_merged().head(3)
    url, params = build_batch_request(first)

    assert url == load_config()["openmeteo"]["base_url"]
    for key in ROW_KEYS:
        assert len(params[key]) == len(first)
    assert params["latitude"] == first["latitude"].tolist()
    assert params["longitude"] == first["longitude"].tolist()
    # acq_date como fecha inicial y final de cada ubicación
    expected_dates = [str(d)[:10] for d in first["acq_date"]]
    assert params["start_date"] == expected_dates
    assert params["end_date"] == expected_dates
    # Los parámetros fijos siguen llegando desde configs/project.yaml
    assert params["hourly"] == load_config()["openmeteo"]["batch_query"]["hourly"]
    assert params["models"] == "best_match"
    assert params["timezone"] == "GMT"
    # batch_rows es interno: nunca viaja a la API
    assert "batch_rows" not in params


@pytest.mark.integration
def test_fetch_next_batch_saves_response():
    """POST por lotes: ``batch_rows`` filas -> respuesta completa en el JSON."""
    batch_rows = load_config()["openmeteo"]["batch_rows"]
    batch = next_batch(load_partial_or_merged())
    assert len(batch) == batch_rows

    # Corta tras guardar la respuesta (horas ponderadas: siguiente paso)
    assert fetch_next_batch() is None

    # La respuesta HTTP queda guardada en JSON para reutilizarla después
    assert OUTPUT_JSON.exists()
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    params = payload["params"]

    # Una respuesta por fila del lote, con sus coordenadas y fechas
    assert len(payload["responses"]) == batch_rows
    for key in ROW_KEYS:
        assert len(params[key]) == batch_rows
    assert params["latitude"] == batch["latitude"].tolist()
    assert params["longitude"] == batch["longitude"].tolist()
    expected_dates = [str(d)[:10] for d in batch["acq_date"]]
    assert params["start_date"] == expected_dates
    assert params["end_date"] == expected_dates

    # Todo en UTC: ni la petición ni las respuestas usan zona horaria local
    assert params["timezone"] == "GMT"
    assert {record["timezone"] for record in payload["responses"]} == {"GMT"}
    assert {record["utc_offset_seconds"] for record in payload["responses"]} == {0}

    # Primera y última fila: coordenadas ≈ pedidas (la API devuelve el centro
    # de la celda), 24 horas desde acq_date y todas las variables pedidas
    hourly_fields = params["hourly"]
    for row, record in (
        (batch.iloc[0], payload["responses"][0]),
        (batch.iloc[-1], payload["responses"][-1]),
    ):
        assert record["latitude"] == pytest.approx(row["latitude"], abs=0.1)
        assert record["longitude"] == pytest.approx(row["longitude"], abs=0.1)
        hours = record["hourly"]
        assert len(hours) == 24
        assert hours[0]["date"].startswith(str(row["acq_date"])[:10])
        assert list(hours[0]) == ["date", *hourly_fields]


# ---------------------------------------------------------------------------
# interpolate_hourly_fields — sin HTTP: se usa la respuesta ya guardada
# ---------------------------------------------------------------------------


def _saved_responses() -> list[dict]:
    """``responses`` de la respuesta HTTP guardada en ``OUTPUT_JSON``."""
    assert OUTPUT_JSON.exists(), f"Missing saved response: {OUTPUT_JSON}"
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))["responses"]


def test_hourly_index_matches_documentation():
    """El índice del hourly equivale a la hora: [0]=00:00, [13]=13:00, [23]=23:00."""
    hourly = _saved_responses()[0]["hourly"]

    assert len(hourly) == 24
    assert hourly[0]["date"].endswith("T00:00:00")
    assert [hourly[i]["date"][11:13] for i in (0, 1, 13, 23)] == [
        "00",
        "01",
        "13",
        "23",
    ]


def test_split_acq_time():
    """``acq_time`` FIRMS (HHMM) -> (hora, minutos, peso de la hora siguiente)."""
    assert split_acq_time(1345) == (13, 45, pytest.approx(0.75))
    assert split_acq_time(221) == (2, 21, pytest.approx(21 / 60))
    assert split_acq_time(221.0) == (2, 21, pytest.approx(0.35))
    assert split_acq_time("1300") == (13, 0, 0.0)
    assert split_acq_time("0000") == (0, 0, 0.0)


def test_interpolate_matches_saved_response():
    """Cada campo del response se interpola con el % de minutos de su fila."""
    responses = _saved_responses()
    first = load_merged().head(3)

    # El JSON guardado tiene que corresponder a las primeras filas (resume=0);
    # cuando la reanudación avance, este fallo avisa de que hay que replantear
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert payload["params"]["latitude"][:3] == first["latitude"].tolist()

    for (_, row), record in zip(first.iterrows(), responses[:3], strict=True):
        hourly = record["hourly"]
        hour, _minutes, pct = split_acq_time(row["acq_time"])
        values = interpolate_hourly_fields(hourly, row["acq_time"])

        # Todas las variables del response, sin la fecha
        assert set(values) == set(hourly[0]) - {"date"}

        # Valor = hora_acq + (hora_sig - hora_acq) * pct
        next_idx = min(hour + 1, len(hourly) - 1)
        for field, value in values.items():
            prev_value = hourly[hour][field]
            next_value = hourly[next_idx][field]
            assert value == pytest.approx(prev_value + (next_value - prev_value) * pct)


def test_interpolate_concrete_value():
    """Ejemplo concreto: fila 0 = acq_time 221 (02:21) -> 35% hacia las 03:00."""
    hourly = _saved_responses()[0]["hourly"]
    row = load_merged().iloc[0]

    assert int(row["acq_time"]) == 221
    values = interpolate_hourly_fields(hourly, row["acq_time"])
    temperatures = [record["temperature_2m"] for record in hourly]

    expected = temperatures[2] + (temperatures[3] - temperatures[2]) * 0.35
    assert values["temperature_2m"] == pytest.approx(expected)


def test_interpolate_edge_cases():
    """Minutos a 0 da el valor exacto; la última hora no se sale del día."""
    synthetic = [
        {"date": f"2023-01-01T{h:02d}:00:00", "temperature_2m": float(h)}
        for h in range(24)
    ]

    # 1300 -> 0%: valor exacto de [13]
    assert interpolate_hourly_fields(synthetic, 1300)["temperature_2m"] == 13.0
    # 1345 -> 75% hacia 14:00 = 13.75 (0.25*13 + 0.75*14)
    assert interpolate_hourly_fields(synthetic, 1345)[
        "temperature_2m"
    ] == pytest.approx(13.75)
    # 2345 -> no existe [24], se usa el valor más cercano: [23]
    assert interpolate_hourly_fields(synthetic, 2345)["temperature_2m"] == 23.0
