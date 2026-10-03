"""Tests del módulo de producción ``wildfire.enrichment.weather_batch``.

Cubren la capa de límites de tasa (clasificación de los 429, cálculo de la
espera hasta el reinicio del contador, ponderación de llamadas) y el bucle
de ejecución ``run_weather_enrichment`` con ``fetch``/``sleep`` inyectados:
sin red y sin dormir de verdad.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from openmeteo_requests.Client import OpenMeteoRequestsError

from wildfire.config import load_config
from wildfire.enrichment.weather_batch import (
    RateLimit,
    WeatherBatchStatus,
    classify_rate_limit,
    completeness_report,
    estimate_weight_range,
    finalize_weather_csv,
    find_resume_row,
    incomplete_rows,
    intersect_weight_ranges,
    load_weather_for_clc,
    pacing_advice,
    run_weather_enrichment,
    seconds_until_reset,
    validate_batch_rows,
    weather_fields,
)

# Motivos reales del servidor (RateLimiter.swift), tal y como los envuelve
# Client._request + weather_api: el cuerpo del 429 queda como texto del
# mensaje de OpenMeteoRequestsError.
MINUTELY_REASON = "Minutely API request limit exceeded. Please try again in one minute."
HOURLY_REASON = "Hourly API request limit exceeded. Please try again in the next hour."
DAILY_REASON = "Daily API request limit exceeded. Please try again tomorrow."
CONCURRENT_REASON = "Too many concurrent requests"
NON_LIMIT_REASON = "Invalid Date"


def _api_error(reason: str) -> OpenMeteoRequestsError:
    """Excepción con la misma forma que produce el SDK para un 400/429."""
    body = {"error": True, "reason": reason}
    return OpenMeteoRequestsError(f"failed to request 'https://x': {body}")


class TestClassifyRateLimit:
    """Los 4 motivos de límite del servidor + lo que NO es límite."""

    @pytest.mark.parametrize(
        ("reason", "expected"),
        [
            (MINUTELY_REASON, RateLimit.MINUTELY),
            (HOURLY_REASON, RateLimit.HOURLY),
            (DAILY_REASON, RateLimit.DAILY),
            (CONCURRENT_REASON, RateLimit.CONCURRENT),
        ],
    )
    def test_server_reasons(self, reason, expected):
        assert classify_rate_limit(_api_error(reason)) is expected

    def test_unknown_future_limit(self):
        """Un límite no reconocido (p. ej. mensual) sigue siendo un límite."""
        reason = "Monthly API request limit exceeded. Please try again next month."
        assert classify_rate_limit(_api_error(reason)) is RateLimit.UNKNOWN_429

    def test_http_400_is_not_a_rate_limit(self):
        """Un 400 (parámetros inválidos) debe re-lanzarse, no esperar."""
        assert classify_rate_limit(_api_error(NON_LIMIT_REASON)) is None

    def test_unrelated_exception(self):
        assert classify_rate_limit(ValueError("boom")) is None


class TestSecondsUntilReset:
    """Espera hasta la frontera UTC + margen de config (reloj inyectado)."""

    def test_minutely_to_next_minute_boundary(self):
        config = load_config()["openmeteo"]
        now = datetime(2026, 6, 15, 12, 0, 30, tzinfo=timezone.utc)

        wait = seconds_until_reset(RateLimit.MINUTELY, now=now)

        assert wait == pytest.approx(30 + config["minutely_buffer_seconds"])

    def test_hourly_to_next_hour_boundary(self):
        config = load_config()["openmeteo"]
        now = datetime(2026, 6, 15, 12, 34, 0, tzinfo=timezone.utc)

        wait = seconds_until_reset(RateLimit.HOURLY, now=now)

        assert wait == pytest.approx(26 * 60 + config["hourly_buffer_seconds"])

    def test_minutely_bounds_with_real_clock(self):
        wait = seconds_until_reset(RateLimit.MINUTELY)

        assert 0 < wait <= 65

    def test_hourly_bounds_with_real_clock(self):
        wait = seconds_until_reset(RateLimit.HOURLY)

        assert 0 < wait <= 3610

    def test_daily_stops_instead_of_waiting(self):
        assert seconds_until_reset(RateLimit.DAILY) is None

    def test_unknown_429_stops_instead_of_waiting(self):
        assert seconds_until_reset(RateLimit.UNKNOWN_429) is None

    def test_concurrent_is_a_short_backoff(self):
        config = load_config()["openmeteo"]

        assert seconds_until_reset(RateLimit.CONCURRENT) == float(
            config["concurrent_backoff_seconds"]
        )


class TestWeightMath:
    """Peso por petición a partir de éxitos previos al 429 y su aplicación."""

    def test_single_success_gives_lower_bound_only(self):
        assert estimate_weight_range(1, 600) == (600.0, float("inf"))

    def test_typical_range(self):
        low, high = estimate_weight_range(12, 600)

        assert low == pytest.approx(50.0)
        assert high == pytest.approx(600 / 11)

    def test_no_successes_is_none(self):
        assert estimate_weight_range(0, 600) is None

    def test_intersection_narrows(self):
        merged = intersect_weight_ranges([(600.0, float("inf")), (600.0, 800.0)])

        assert merged == (600.0, 800.0)

    def test_disjoint_ranges_are_none(self):
        assert intersect_weight_ranges([(100.0, 200.0), (300.0, 400.0)]) is None

    def test_empty_ranges_are_none(self):
        assert intersect_weight_ranges([]) is None

    def test_pacing_advice_for_weight_750(self):
        advice = pacing_advice(750)

        # ceil(600/750)=1/min, ceil(5000/750)=7/h, ceil(10000/750)=14/día
        assert advice["per_minute"] == 1
        assert advice["per_hour"] == 7
        assert advice["per_day"] == 14
        # Manda el promedio horario (3600/7), no el minuto (60/1)
        assert advice["interval_seconds"] == pytest.approx(3600 / 7)

    def test_pacing_advice_for_light_weight(self):
        advice = pacing_advice(50)

        assert advice["per_minute"] == 12
        assert advice["per_hour"] == 100
        assert advice["per_day"] == 200
        assert advice["interval_seconds"] == pytest.approx(36)

    def test_retry_config_keys_stay_out_of_batch_query(self):
        """Las claves de reintento no viajan a la API (batch_query es verbatim)."""
        openmeteo = load_config()["openmeteo"]

        for key in (
            "batch_interval_seconds",
            "minutely_buffer_seconds",
            "hourly_buffer_seconds",
            "concurrent_backoff_seconds",
            "max_consecutive_waits",
        ):
            assert key in openmeteo
            assert key not in openmeteo["batch_query"]


# ---------------------------------------------------------------------------
# run_weather_enrichment — bucle con fetch/sleep inyectados, sin red
# ---------------------------------------------------------------------------


def _make_csv(tmp_path: Path, rows: int = 3) -> Path:
    """CSV de reanudación mínimo sin columnas de clima (resume = 0)."""
    df = pd.DataFrame(
        {
            "latitude": [40.0] * rows,
            "acq_date": ["2023-01-01"] * rows,
            "acq_time": [1200] * rows,
        }
    )
    path = tmp_path / "partial.csv"
    df.to_csv(path, index=False)
    return path


def _make_fetch(outcomes: list[BaseException | None]):
    """Fetch falso: ``None`` = lote correcto que rellena la fila de reanudación.

    Cada entrada de ``outcomes`` se consume en orden (se relanza la excepción);
    agotada la lista, todos los lotes restantes son correctos. Las llamadas
    se cuentan en ``fake.calls``.
    """
    state = {"calls": 0}

    def fake(*, save_path, csv_path):
        index = state["calls"]
        state["calls"] += 1
        outcome = outcomes[index] if index < len(outcomes) else None
        if outcome is not None:
            raise outcome
        df = pd.read_csv(csv_path)
        if "temperature_2m" not in df.columns:
            df["temperature_2m"] = float("nan")
        resume = find_resume_row(df)
        if resume < len(df):
            df.iloc[resume, df.columns.get_loc("temperature_2m")] = 1.0
            df.to_csv(csv_path, index=False)

    fake.calls = 0

    def wrapper(*, save_path, csv_path):
        fake.calls += 1
        return fake(save_path=save_path, csv_path=csv_path)

    wrapper.calls = state  # dict con "calls"; se lee como wrapper.calls["calls"]
    return wrapper


def _run(tmp_path, csv, outcomes, **kwargs):
    """``run_weather_enrichment`` sobre ``csv`` con fetch/sleep de mentira."""
    sleeps: list[float] = []
    defaults = {
        "csv_path": csv,
        "save_path": tmp_path / "response.json",
        "fetch": _make_fetch(outcomes),
        "sleep": sleeps.append,
        "interval_seconds": 0,
    }
    defaults.update(kwargs)
    status = run_weather_enrichment(**defaults)
    return status, defaults["fetch"], sleeps


class TestRunWeatherEnrichment:
    """El bucle: éxito + pacing, reintentos por límite, paradas limpias."""

    def test_completes_all_batches_with_pacing(self, tmp_path):
        csv = _make_csv(tmp_path, rows=3)

        status, fetch, sleeps = _run(tmp_path, csv, [None] * 3, interval_seconds=0.5)

        assert status is WeatherBatchStatus.COMPLETE
        assert fetch.calls["calls"] == 3
        # Pausa tras cada lote correcto salvo el último (antes de comprobar
        # que ya no queda nada)
        assert sleeps == [0.5, 0.5]

    def test_already_complete_never_fetches(self, tmp_path):
        csv = _make_csv(tmp_path, rows=2)
        pd.DataFrame({"temperature_2m": [1.0, 2.0]}).to_csv(csv, index=False)

        status, fetch, sleeps = _run(tmp_path, csv, [])

        assert status is WeatherBatchStatus.COMPLETE
        assert fetch.calls["calls"] == 0
        assert sleeps == []

    def test_minutely_429_waits_and_retries_same_rows(self, tmp_path):
        csv = _make_csv(tmp_path, rows=2)
        error = _api_error(MINUTELY_REASON)

        status, fetch, sleeps = _run(tmp_path, csv, [error, error, None, None])

        assert status is WeatherBatchStatus.COMPLETE
        assert fetch.calls["calls"] == 4
        assert len(sleeps) == 2
        # Espera de minuto: hasta la frontera UTC + margen (nunca 0, <= 65 s)
        assert all(0 < wait <= 65 for wait in sleeps)
        # La reanudación no avanzó con los 429: 2 lotes correctos rellenan 2 filas
        df = pd.read_csv(csv)
        assert find_resume_row(df) == 2

    def test_hourly_429_waits_for_hour_reset(self, tmp_path):
        csv = _make_csv(tmp_path, rows=1)
        error = _api_error(HOURLY_REASON)

        status, fetch, sleeps = _run(tmp_path, csv, [error, None])

        assert status is WeatherBatchStatus.COMPLETE
        assert fetch.calls["calls"] == 2
        assert len(sleeps) == 1
        assert 0 < sleeps[0] <= 3610

    def test_daily_429_stops_cleanly_without_touching_csv(self, tmp_path):
        csv = _make_csv(tmp_path, rows=3)

        status, fetch, sleeps = _run(tmp_path, csv, [_api_error(DAILY_REASON)])

        assert status is WeatherBatchStatus.DAILY_STOP
        assert fetch.calls["calls"] == 1
        assert sleeps == []
        # Sin columnas de clima: la reanudación queda donde estaba
        assert find_resume_row(pd.read_csv(csv)) == 0

    def test_unknown_limit_stops_cleanly(self, tmp_path):
        csv = _make_csv(tmp_path, rows=1)
        error = _api_error("Monthly API request limit exceeded.")

        status, _fetch, _sleeps = _run(tmp_path, csv, [error])

        assert status is WeatherBatchStatus.DAILY_STOP

    def test_non_limit_error_propagates(self, tmp_path):
        csv = _make_csv(tmp_path, rows=1)

        with pytest.raises(OpenMeteoRequestsError):
            _run(tmp_path, csv, [_api_error(NON_LIMIT_REASON)])

    def test_unrelated_exception_propagates(self, tmp_path):
        csv = _make_csv(tmp_path, rows=1)

        with pytest.raises(ValueError):
            _run(tmp_path, csv, [ValueError("boom")])

    def test_wait_cap_stops_after_consecutive_limits(self, tmp_path):
        csv = _make_csv(tmp_path, rows=3)
        error = _api_error(MINUTELY_REASON)

        status, fetch, sleeps = _run(
            tmp_path,
            csv,
            [error] * 4,
            max_consecutive_waits=2,
        )

        assert status is WeatherBatchStatus.WAIT_CAP
        # 1º límite -> espera, 2º límite -> espera, 3º límite -> se rinde
        assert fetch.calls["calls"] == 3
        assert len(sleeps) == 2

    def test_success_resets_consecutive_wait_counter(self, tmp_path):
        """Un lote correcto entre medias reinicia el contador de esperas."""
        csv = _make_csv(tmp_path, rows=4)
        error = _api_error(MINUTELY_REASON)

        status, fetch, sleeps = _run(
            tmp_path,
            csv,
            [error, None, error, None, None, None],
            max_consecutive_waits=1,
        )

        assert status is WeatherBatchStatus.COMPLETE
        assert fetch.calls["calls"] == 6
        assert len(sleeps) == 2

    def test_max_batches_caps_the_run(self, tmp_path):
        csv = _make_csv(tmp_path, rows=5)

        status, fetch, sleeps = _run(tmp_path, csv, [None] * 5, max_batches=2)

        assert status is WeatherBatchStatus.MAX_BATCHES
        assert fetch.calls["calls"] == 2
        assert sleeps == []
        assert find_resume_row(pd.read_csv(csv)) == 2


# ---------------------------------------------------------------------------
# Celda 3 del notebook: load_weather_for_clc + finalize_weather_csv
# ---------------------------------------------------------------------------


def _weather_csv(path: Path, rows: int, filled: int) -> Path:
    """CSV con las primeras ``filled`` de ``rows`` filas con temperature_2m."""
    df = pd.DataFrame(
        {
            "latitude": [40.0] * rows,
            "temperature_2m": [10.0] * filled + [float("nan")] * (rows - filled),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _full_weather_csv(path: Path, rows: int, gaps: dict | None = None) -> Path:
    """CSV con las 15 variables rellenas; ``gaps`` = {col: [índices a NaN]}."""
    gaps = gaps or {}
    df = pd.DataFrame({"latitude": [40.0] * rows})
    for field in weather_fields():
        values = [10.0] * rows
        for i in gaps.get(field, []):
            values[i] = float("nan")
        df[field] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


class TestLoadWeatherForClc:
    """Terminado si existe; si no, parcial filtrado a filas con clima."""

    def test_completed_wins_over_partial(self, tmp_path):
        completed = _weather_csv(tmp_path / "weather.csv", rows=4, filled=4)
        _weather_csv(tmp_path / "weather_partial.csv", rows=6, filled=2)

        out = load_weather_for_clc(completed, tmp_path / "weather_partial.csv")

        assert len(out) == 4  # el parcial completo se ignora
        assert out["temperature_2m"].notna().all()

    def test_falls_back_to_partial_and_filters_unfilled(self, tmp_path):
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=6, filled=4)
        missing = tmp_path / "weather.csv"

        out = load_weather_for_clc(missing, partial)

        assert len(out) == 4  # solo las filas ya descargadas
        assert out["temperature_2m"].notna().all()
        # conserva el orden original del parcial
        assert out["latitude"].tolist() == [40.0] * 4

    def test_result_has_clean_positional_index(self, tmp_path):
        df = pd.DataFrame(
            {
                "latitude": [40.0] * 4,
                "temperature_2m": [float("nan"), float("nan"), 10.0, 11.0],
            }
        )
        partial = tmp_path / "weather_partial.csv"
        df.to_csv(partial, index=False)

        out = load_weather_for_clc(tmp_path / "weather.csv", partial)

        assert out.index.tolist() == [0, 1]  # no conserva [2, 3]
        assert out["temperature_2m"].tolist() == [10.0, 11.0]

    def test_no_weather_column_raises(self, tmp_path):
        partial = _make_csv(tmp_path, rows=3)  # columnas FIRMS sin clima
        missing = tmp_path / "weather.csv"

        with pytest.raises(RuntimeError, match="Open-Meteo step"):
            load_weather_for_clc(missing, partial)

    def test_all_rows_unfilled_raises(self, tmp_path):
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=3, filled=0)
        missing = tmp_path / "weather.csv"

        with pytest.raises(RuntimeError, match="Open-Meteo step"):
            load_weather_for_clc(missing, partial)


class TestIncompleteRows:
    """Filas con temperature_2m pero alguna variable en NaN."""

    def test_temp_only_csv_is_all_incomplete(self, tmp_path):
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=3, filled=3)
        df = pd.read_csv(partial)

        assert len(incomplete_rows(df)) == 3

    def test_full_csv_has_no_incomplete(self, tmp_path):
        partial = _full_weather_csv(tmp_path / "weather_partial.csv", rows=3)
        df = pd.read_csv(partial)

        assert incomplete_rows(df).empty

    def test_single_variable_gap_detected(self, tmp_path):
        partial = _full_weather_csv(
            tmp_path / "weather_partial.csv",
            rows=4,
            gaps={"boundary_layer_height": [1, 3]},
        )
        df = pd.read_csv(partial)

        out = incomplete_rows(df)
        assert out.index.tolist() == [1, 3]

    def test_no_temperature_column_is_empty(self):
        df = pd.DataFrame({"latitude": [40.0]})
        assert incomplete_rows(df).empty

    def test_completeness_report_counts(self, tmp_path):
        partial = _full_weather_csv(
            tmp_path / "weather_partial.csv",
            rows=4,
            gaps={"boundary_layer_height": [0, 2]},
        )
        df = pd.read_csv(partial)

        report = completeness_report(df)
        assert report["rows_with_weather"] == 4
        assert report["boundary_layer_height"] == 2
        assert report["temperature_2m"] == 0
        assert report["incomplete_rows"] == 2


class TestValidateBatchRows:
    def _batch(self, **overrides):
        base = {
            "latitude": [40.0],
            "longitude": [-3.0],
            "acq_date": ["2023-07-01"],
            "acq_time": [1200],
        }
        base.update(overrides)
        return pd.DataFrame(base)

    def test_valid_batch_has_no_errors(self):
        assert validate_batch_rows(self._batch()) == []

    def test_nan_coordinates_reported_with_index(self):
        df = self._batch()
        df.index = [501]
        df.loc[501, "latitude"] = float("nan")
        errors = validate_batch_rows(df)
        assert len(errors) == 1
        assert "501" in errors[0] and "latitude" in errors[0]

    def test_bad_date_reported(self):
        errors = validate_batch_rows(self._batch(acq_date=["ayer"]))
        assert len(errors) == 1
        assert "acq_date" in errors[0]

    def test_bad_time_reported(self):
        assert validate_batch_rows(self._batch(acq_time=["2400"]))
        assert validate_batch_rows(self._batch(acq_time=[None]))
        assert validate_batch_rows(self._batch()) == []

    def test_multiple_problems_all_reported(self):
        df = pd.DataFrame(
            {
                "latitude": [float("nan"), 40.0],
                "longitude": [-3.0, -3.0],
                "acq_date": ["2023-07-01", "nope"],
                "acq_time": [1200, 1200],
            }
        )
        assert len(validate_batch_rows(df)) == 2


class TestFinalizeWeatherCsv:
    """Copia parcial→terminado solo cuando la reanudación está al final."""

    def test_copies_when_partial_is_complete(self, tmp_path):
        partial = _full_weather_csv(tmp_path / "weather_partial.csv", rows=3)
        completed = tmp_path / "weather.csv"

        result = finalize_weather_csv(partial, completed)

        assert result == completed
        pd.testing.assert_frame_equal(pd.read_csv(completed), pd.read_csv(partial))

    def test_temp_only_partial_is_not_publishable_by_default(self, tmp_path):
        # 15 columnas ausentes = hueco: con require_complete no se publica.
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=3, filled=3)
        completed = tmp_path / "weather.csv"

        result = finalize_weather_csv(partial, completed)

        assert result is None
        assert not completed.exists()

    def test_require_complete_false_keeps_legacy_sentinel(self, tmp_path):
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=3, filled=3)
        completed = tmp_path / "weather.csv"

        result = finalize_weather_csv(partial, completed, require_complete=False)

        assert result == completed
        assert completed.exists()

    def test_partial_variable_gap_blocks_publish(self, tmp_path):
        partial = _full_weather_csv(
            tmp_path / "weather_partial.csv",
            rows=3,
            gaps={"boundary_layer_height": [2]},
        )
        completed = tmp_path / "weather.csv"

        result = finalize_weather_csv(partial, completed)

        assert result is None
        assert not completed.exists()

    def test_incomplete_partial_creates_nothing(self, tmp_path):
        partial = _weather_csv(tmp_path / "weather_partial.csv", rows=6, filled=4)
        completed = tmp_path / "weather.csv"

        result = finalize_weather_csv(partial, completed)

        assert result is None
        assert not completed.exists()

    def test_idempotent_when_completed_exists(self, tmp_path):
        partial = _full_weather_csv(tmp_path / "weather_partial.csv", rows=3)
        completed = _full_weather_csv(tmp_path / "weather.csv", rows=3)
        completed.write_text("sentinel\n", encoding="utf-8")

        result = finalize_weather_csv(partial, completed)

        assert result == completed
        assert completed.read_text(encoding="utf-8") == "sentinel\n"  # sin copia

    def test_missing_partial_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Partial CSV not found"):
            finalize_weather_csv(tmp_path / "nope.csv", tmp_path / "weather.csv")
