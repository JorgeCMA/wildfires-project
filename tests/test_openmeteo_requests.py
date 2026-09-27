"""Integration test for the ``openmeteo-requests`` SDK (batch multi-location query)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from wildfire.config import PROJECT_ROOT, load_config
from wildfire.enrichment import load_latest_merged_split

# Claves cuyo valor varía por fila: se convierten en listas (una entrada por
# ubicación) para la consulta por lotes. El resto (hourly, models, timezone)
# vive en configs/project.yaml bajo ``openmeteo.batch_query``.
ROW_KEYS = ("latitude", "longitude", "start_date", "end_date")

# Dónde se guarda la respuesta HTTP para reutilizarla después
OUTPUT_JSON = (
    PROJECT_ROOT / load_config()["output"]["processed"] / "openmeteo_response.json"
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


def build_batch_request(df: pd.DataFrame | None = None) -> tuple[str, dict]:
    """Construye ``(url, params)`` de la consulta por lotes multi-ubicación.

    Latitud, longitud y fechas salen de las filas de entrada: cada una aporta
    su ``latitude``/``longitude`` y su ``acq_date`` como ``start_date`` y
    ``end_date``. Por defecto se usan las 3 primeras filas del merged más
    reciente (``load_latest_merged_split``). El resto de parámetros
    (``hourly``, ``models``, ``timezone``) viene de ``configs/project.yaml``.
    """
    if df is None:
        df, _ = load_latest_merged_split()

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


def run_multi_location_query(
    save_path: Path | None = OUTPUT_JSON,
) -> list[pd.DataFrame]:
    """Ejecuta una consulta por lotes a Open-Meteo para varias ubicaciones.

    Los parámetros los construye ``build_batch_request`` (3 primeras filas del
    merged). Devuelve un ``DataFrame`` por ubicación con las variables
    horarias pedidas.

    Parameters
    ----------
    save_path:
        Si no es ``None``, guarda la respuesta HTTP como JSON (petición +
        un registro por ubicación) en esta ruta, sobrescribiéndola.
    """
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    # Cliente de la API con caché y reintentos (el código de abajo asume `openmeteo`)
    session = requests_cache.CachedSession(".cache", expire_after=3600)
    session = retry(session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=session)

    url, params = build_batch_request()

    responses = openmeteo.weather_api(url, params=params)

    dataframes: list[pd.DataFrame] = []
    records: list[dict] = []

    # Process each location
    for response in responses:
        # Con timezone=GMT el flatbuffer no trae nombre de zona horaria
        timezone = (response.Timezone() or b"GMT").decode()

        print(f"\nCoordinates: {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation: {response.Elevation()} m asl")
        print(f"Timezone: {timezone}")
        print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_surface_pressure = hourly.Variables(2).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()
        hourly_wind_gusts_10m = hourly.Variables(6).ValuesAsNumpy()
        hourly_rain = hourly.Variables(7).ValuesAsNumpy()
        hourly_et0_fao_evapotranspiration = hourly.Variables(8).ValuesAsNumpy()
        hourly_shortwave_radiation = hourly.Variables(9).ValuesAsNumpy()
        hourly_boundary_layer_height = hourly.Variables(10).ValuesAsNumpy()
        hourly_is_day = hourly.Variables(11).ValuesAsNumpy()
        hourly_vapour_pressure_deficit = hourly.Variables(12).ValuesAsNumpy()
        hourly_soil_moisture_0_to_7cm = hourly.Variables(13).ValuesAsNumpy()
        hourly_soil_moisture_7_to_28cm = hourly.Variables(14).ValuesAsNumpy()

        # El epoch del flatbuffer es UTC absoluto y con timezone=GMT el offset
        # es 0, así que la ventana pedida sale directa: 00:00-23:00 de
        # acq_date en UTC (igual que acq_date/acq_time de FIRMS).
        assert response.UtcOffsetSeconds() == 0
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["surface_pressure"] = hourly_surface_pressure
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
        hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
        hourly_data["rain"] = hourly_rain
        hourly_data["et0_fao_evapotranspiration"] = hourly_et0_fao_evapotranspiration
        hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
        hourly_data["boundary_layer_height"] = hourly_boundary_layer_height
        hourly_data["is_day"] = hourly_is_day
        hourly_data["vapour_pressure_deficit"] = hourly_vapour_pressure_deficit
        hourly_data["soil_moisture_0_to_7cm"] = hourly_soil_moisture_0_to_7cm
        hourly_data["soil_moisture_7_to_28cm"] = hourly_soil_moisture_7_to_28cm

        hourly_dataframe = pd.DataFrame(data=hourly_data)
        print("\nHourly data\n", hourly_dataframe)

        dataframes.append(hourly_dataframe)

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
        print(f"\nSaved HTTP response to: {save_path}")

    return dataframes


def test_row_requests_from_first_rows():
    """Las 3 primeras filas del merged generan 3 peticiones con su propia fecha."""
    first, _ = load_latest_merged_split()
    row_requests = build_row_requests(first)

    assert len(row_requests) == 3
    for req, (_, row) in zip(row_requests, first.iterrows(), strict=True):
        assert req["latitude"] == row["latitude"]
        assert req["longitude"] == row["longitude"]
        assert req["start_date"] == str(row["acq_date"])[:10]
        assert req["end_date"] == req["start_date"]
        assert req["hourly"] == load_config()["openmeteo"]["batch_query"]["hourly"]


def test_build_batch_request_from_first_rows():
    """La consulta por lotes toma lat/lon/fechas de las 3 primeras filas."""
    first, _ = load_latest_merged_split()
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


@pytest.mark.integration
def test_multi_location_batch_query():
    """Una sola petición devuelve las 3 ubicaciones con 24 horas y 15 variables."""
    first, _ = load_latest_merged_split()
    url, params = build_batch_request(first)
    dataframes = run_multi_location_query()

    # La respuesta HTTP queda guardada en JSON para reutilizarla después
    assert OUTPUT_JSON.exists()
    payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
    assert payload["url"] == url
    assert payload["params"] == params
    # Todo en UTC: ni la petición ni las respuestas usan zona horaria local
    assert payload["params"]["timezone"] == "GMT"
    assert {record["timezone"] for record in payload["responses"]} == {"GMT"}
    assert {record["utc_offset_seconds"] for record in payload["responses"]} == {0}

    # Una respuesta por cada par (lat, lon) pedido
    assert len(dataframes) == len(params["latitude"])
    assert len(params["latitude"]) == len(params["longitude"])

    for df in dataframes:
        # Un día completo de datos horarios (acq_date == start_date == end_date)
        assert len(df) == 24
        # date + las variables pedidas, en el mismo orden
        assert list(df.columns) == ["date", *params["hourly"]]
        # Sin huecos en la variable principal y horario continuo de 1h
        assert df["temperature_2m"].notna().all()
        assert df["date"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
        # is_day es binario
        assert set(df["is_day"].unique()) <= {0.0, 1.0}

    # Cada registro corresponde a su fila: coordenadas ≈ pedidas (la API
    # devuelve el centro de la celda más cercana) y fecha = acq_date
    for (_, row), record in zip(first.iterrows(), payload["responses"], strict=True):
        assert record["latitude"] == pytest.approx(row["latitude"], abs=0.1)
        assert record["longitude"] == pytest.approx(row["longitude"], abs=0.1)
        hours = record["hourly"]
        assert len(hours) == 24
        assert hours[0]["date"].startswith(str(row["acq_date"])[:10])


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
    first, _ = load_latest_merged_split()

    for (_, row), record in zip(first.iterrows(), responses, strict=True):
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
    first, _ = load_latest_merged_split()
    row = first.iloc[0]

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
