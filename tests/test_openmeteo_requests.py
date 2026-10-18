"""Tests de la descarga por lotes ``openmeteo-requests``: reanudación y 429.

La implementación vive en ``wildfire.enrichment.weather_batch`` (módulo de
producción): aquí se prueba la configuración y la reanudación, el relleno
del CSV parcial, la construcción de peticiones, la interpolación horaria y
el POST real (marcado ``integration``). La respuesta HTTP no se guarda en
disco: los tests de interpolación usan un payload sintético en memoria.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wildfire.config import load_config
from wildfire.enrichment import load_merged
from wildfire.enrichment.weather_batch import (
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
def test_fetch_next_batch_fills_partial(tmp_path):
    """POST por lotes: ``batch_rows`` filas rellenadas en el CSV parcial."""
    from openmeteo_requests.Client import OpenMeteoRequestsError

    batch_rows = load_config()["openmeteo"]["batch_rows"]
    csv_path = tmp_path / "partial.csv"

    # Sin CSV previo la entrada es el merged y el resume empieza en 0
    assert not csv_path.exists()
    batch = next_batch(load_partial_or_merged(csv_path))
    assert 0 < len(batch) <= batch_rows

    try:
        assert fetch_next_batch(csv_path=csv_path) is None
    except OpenMeteoRequestsError as exc:
        # Cuota gastada por la descarga en paralelo: skip, no fail.
        if "429" in str(exc) or "limit" in str(exc).lower():
            pytest.skip(f"Open-Meteo quota exhausted: {exc}")
        raise

    # CSV parcial: merged + 15 columnas de clima, lote relleno, resto sin tocar
    df = pd.read_csv(csv_path)
    assert list(df.columns) == [*load_merged().columns, *weather_fields()]
    assert len(df) == len(load_merged())
    n = len(batch)
    assert df["temperature_2m"].iloc[:n].notna().all()
    assert df["temperature_2m"].iloc[n:].isna().all()
    assert find_resume_row(df) == n


# ---------------------------------------------------------------------------
# interpolate_hourly_fields — sin HTTP: payload sintético en memoria
# (la respuesta HTTP real ya no se guarda en disco)
# ---------------------------------------------------------------------------


def _synthetic_payload(rows: pd.DataFrame) -> dict:
    """Payload con la misma forma que la respuesta de la API, generado en memoria.

    Un registro por fila, hourly de 24 horas con las 15 variables: valores
    deterministas distintos por hora/variable/fila para que la interpolación
    no sea trivial.
    """
    hourly_fields = weather_fields()
    responses = []
    for row_idx, _row in rows.iterrows():
        hourly = []
        for hour in range(24):
            record = {"date": f"2023-06-15T{hour:02d}:00:00"}
            for pos, field in enumerate(hourly_fields):
                record[field] = round(hour * (pos + 1) + row_idx, 3)
            hourly.append(record)
        responses.append({"timezone": "GMT", "utc_offset_seconds": 0, "hourly": hourly})
    return {
        "row_start": 0,
        "params": {
            "latitude": rows["latitude"].tolist(),
            "longitude": rows["longitude"].tolist(),
        },
        "responses": responses,
    }


def _saved_responses() -> list[dict]:
    """``responses`` del payload sintético de las 3 primeras filas del merged."""
    return _synthetic_payload(load_merged().head(3))["responses"]


def _saved_rows() -> pd.DataFrame:
    """Filas del merged a las que corresponde el payload sintético."""
    return load_merged().head(3)


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


def test_interpolate_matches_synthetic_response():
    """Cada campo del response se interpola con el % de minutos de su fila."""
    responses = _saved_responses()
    rows = _saved_rows().head(3)

    for (_, row), record in zip(rows.iterrows(), responses, strict=True):
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
