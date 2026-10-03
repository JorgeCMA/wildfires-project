"""Tests de la descarga por lotes ``openmeteo-requests``: reanudación y 429.

La implementación vive en ``wildfire.enrichment.weather_batch`` (módulo de
producción): aquí se prueba la configuración y la reanudación, el relleno
del CSV parcial, la construcción de peticiones, la interpolación horaria y
el POST real (marcado ``integration``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from wildfire.config import load_config
from wildfire.enrichment import load_merged
from wildfire.enrichment.weather_batch import (
    OUTPUT_JSON,
    ROW_KEYS,
    build_batch_request,
    build_row_requests,
    fetch_next_batch,
    fill_weather,
    find_resume_row,
    interpolate_hourly_fields,
    load_partial_or_merged,
    next_batch,
    split_acq_time,
    weather_fields,
)

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


# ---------------------------------------------------------------------------
# fill_weather — relleno del CSV parcial, sin HTTP
# ---------------------------------------------------------------------------


def _synthetic_record(base_value: float) -> dict:
    """Registro con 24 horas: cada variable vale ``base_value + hora``."""
    fields = weather_fields()
    hourly = []
    for hour in range(24):
        entry: dict = {"date": f"2023-01-01T{hour:02d}:00:00"}
        entry.update({field: base_value + hour for field in fields})
        hourly.append(entry)
    return {"hourly": hourly}


def test_fill_weather_adds_columns_and_values():
    """Crea las 15 columnas y escribe el valor interpolado por fila."""
    df = pd.DataFrame({"acq_time": [1345, 0]})
    records = [_synthetic_record(0.0), _synthetic_record(100.0)]

    fill_weather(df, 0, records)

    assert list(df.columns) == ["acq_time", *weather_fields()]
    # 1345 -> 25% de 13:00 + 75% de 14:00 con valores 13/14 (base 0)
    assert df.loc[0, "temperature_2m"] == pytest.approx(13.75)
    # 0000 -> valor exacto de 00:00 con base 100 (detecta desalineación)
    assert df.loc[1, "temperature_2m"] == pytest.approx(100.0)


def test_fill_weather_leaves_tail_nan_and_advances_resume():
    """Solo se rellena el lote: el resto queda NaN y el resume avanza."""
    df = pd.DataFrame({"acq_time": [1300] * 4})

    fill_weather(df, 0, [_synthetic_record(0.0), _synthetic_record(1.0)])
    assert df["temperature_2m"].iloc[:2].notna().all()
    assert df["temperature_2m"].iloc[2:].isna().all()
    assert find_resume_row(df) == 2

    fill_weather(df, 2, [_synthetic_record(2.0), _synthetic_record(3.0)])
    assert df["temperature_2m"].notna().all()
    assert find_resume_row(df) == 4


def test_fill_weather_handles_none_values():
    """Un extremo ``None`` (NaN de la API) deja la fila sin dato, sin crashear."""
    record = _synthetic_record(0.0)
    record["hourly"][13]["temperature_2m"] = None
    df = pd.DataFrame({"acq_time": [1345]})

    fill_weather(df, 0, [record])

    # 1345 usa [13] (None) y [14] -> None -> NaN en el CSV
    assert pd.isna(df.loc[0, "temperature_2m"])
    # Las demás variables del mismo registro siguen rellenándose
    assert df.loc[0, "precipitation"] == pytest.approx(13.75)


def test_fill_weather_length_mismatch_raises():
    """Respuesta con distinto número de registros que filas: error antes de escribir."""
    df = pd.DataFrame({"acq_time": [1300]})
    record = _synthetic_record(0.0)

    with pytest.raises(ValueError):
        fill_weather(df, 0, [record, record])


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
def test_fetch_next_batch_saves_response(tmp_path):
    """POST por lotes: ``batch_rows`` filas -> JSON + CSV parcial rellenado."""
    from openmeteo_requests.Client import OpenMeteoRequestsError

    batch_rows = load_config()["openmeteo"]["batch_rows"]
    csv_path = tmp_path / "partial.csv"

    # Sin CSV previo la entrada es el merged y el resume empieza en 0
    assert not csv_path.exists()
    batch = next_batch(load_partial_or_merged(csv_path))
    assert 0 < len(batch) <= batch_rows

    # Corta tras JSON + CSV (ambos por defecto: OUTPUT_JSON / tmp aquí).
    # save_path explícito: por defecto escribiría en el JSON rastreado
    # data/processed/openmeteo_response.json.
    save_path = tmp_path / "response.json"
    try:
        assert fetch_next_batch(csv_path=csv_path, save_path=save_path) is None
    except OpenMeteoRequestsError as exc:
        # Cuota gastada por la descarga en paralelo: skip, no fail.
        if "429" in str(exc) or "limit" in str(exc).lower():
            pytest.skip(f"Open-Meteo quota exhausted: {exc}")
        raise

    # La respuesta HTTP queda guardada en JSON para reutilizarla después
    assert save_path.exists()
    payload = json.loads(save_path.read_text(encoding="utf-8"))
    url, params = build_batch_request(batch)
    assert payload["row_start"] == 0
    assert payload["url"] == url
    assert payload["params"] == params
    assert len(payload["responses"]) == len(batch)

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

    # CSV parcial: merged + 15 columnas de clima, lote relleno, resto sin tocar
    df = pd.read_csv(csv_path)
    assert list(df.columns) == [*load_merged().columns, *hourly_fields]
    assert len(df) == len(load_merged())
    n = len(batch)
    assert df["temperature_2m"].iloc[:n].notna().all()
    assert df["temperature_2m"].iloc[n:].isna().all()
    assert find_resume_row(df) == n

    # La fila 0 del CSV coincide con la interpolación de su respuesta
    values = interpolate_hourly_fields(
        payload["responses"][0]["hourly"], df.loc[0, "acq_time"]
    )
    assert df.loc[0, "temperature_2m"] == pytest.approx(values["temperature_2m"])


# ---------------------------------------------------------------------------
# interpolate_hourly_fields — sin HTTP: se usa la respuesta ya guardada
# ---------------------------------------------------------------------------


def _payload() -> dict:
    """Respuesta HTTP guardada en ``OUTPUT_JSON``."""
    assert OUTPUT_JSON.exists(), f"Missing saved response: {OUTPUT_JSON}"
    return json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))


def _saved_responses() -> list[dict]:
    """``responses`` de la respuesta HTTP guardada en ``OUTPUT_JSON``."""
    return _payload()["responses"]


def _saved_rows() -> pd.DataFrame:
    """Filas del CSV (parcial o merged) a las que corresponde la respuesta."""
    payload = _payload()
    rows = load_partial_or_merged().iloc[payload["row_start"] :]
    return rows.iloc[: len(payload["responses"])]


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
    payload = _payload()
    responses = payload["responses"]
    rows = _saved_rows().head(3)

    # El JSON y las filas tienen que corresponder (mismo row_start, mismo orden)
    assert payload["params"]["latitude"][: len(rows)] == rows["latitude"].tolist()
    assert payload["params"]["longitude"][: len(rows)] == rows["longitude"].tolist()

    for (_, row), record in zip(rows.iterrows(), responses[: len(rows)], strict=True):
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
            if prev_value is None or next_value is None:
                assert value is None
                continue
            assert value == pytest.approx(prev_value + (next_value - prev_value) * pct)


def test_interpolate_concrete_value():
    """Fila 0 del lote guardado: valor interpolado exacto con su acq_time."""
    hourly = _saved_responses()[0]["hourly"]
    row = _saved_rows().iloc[0]
    values = interpolate_hourly_fields(hourly, row["acq_time"])
    temperatures = [record["temperature_2m"] for record in hourly]

    hour, _minutes, pct = split_acq_time(row["acq_time"])
    next_idx = min(hour + 1, len(hourly) - 1)
    expected = temperatures[hour] + (temperatures[next_idx] - temperatures[hour]) * pct
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
