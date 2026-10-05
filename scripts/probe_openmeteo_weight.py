"""Sonda del peso real de nuestras peticiones por lotes en Open-Meteo.

El limitador gratuito de Open-Meteo (600/min, 5.000/h, 10.000/día) cuenta
llamadas *ponderadas*, no peticiones: una petición de S ubicaciones × 15
variables pesa cientos de unidades (el peso crece linealmente con las
ubicaciones). Esta sonda lo mide empíricamente, sin gastar la caché de
``requests-cache`` (así se ve el gasto real del servidor):

1. Espera al comienzo de un minuto UTC: el contador de minuto arranca en 0
   y el ``check`` del servidor ocurre antes del ``increment``, así que la
   primera petición de la ventana siempre se acepta.
2. Lanza la misma petición repetidamente hasta recibir 429 (o un máximo de
   ``--max-attempts`` éxitos). Con N éxitos y la N+1 en 429:
   ``N·peso >= límite`` y ``(N-1)·peso < límite`` -> peso en [límite/N,
   límite/(N-1)).
3. Si N es pequeño (resolución pobre) se repite con un cuerpo más pequeño
   (S // 4) y se escalan los rangos a ``batch_rows`` (peso lineal con S),
   intersecándolos.

Coste típico: 1-3 rondas, cada una <= ~750 ponderadas (<= 7,5 % del
presupuesto diario de 10.000). Si el resultado lo permite, un lote completo
no vuelve a necesitar esta sonda.

Salida: rango de peso por lote, llamadas por ventana previstas y el
``batch_interval_seconds`` sugerido para ``configs/project.yaml``.

Uso::

    python scripts/probe_openmeteo_weight.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from wildfire.config import load_config
from wildfire.enrichment.merge_sensors import load_merged
from wildfire.enrichment.weather_batch import (
    RateLimit,
    build_batch_request,
    classify_reason,
    estimate_weight_range,
    intersect_weight_ranges,
    pacing_advice,
)

# Límites por ventana (tier gratuito), en llamadas ponderadas.
_WINDOW_LIMITS = {
    RateLimit.MINUTELY: 600,
    RateLimit.HOURLY: 5000,
    RateLimit.DAILY: 10000,
}

# Resultados de ``fire`` que impiden sondear ahora (presupuesto/agotado).
_ABORT_RESULTS = ("hourly", "daily", "unknown_429", "concurrent")


def fire(session: requests.Session, url: str, params: dict) -> tuple[str, str]:
    """POST sin caché ni reintentos de transporte.

    Returns
    -------
    tuple[str, str]
        ``(resultado, motivo)`` donde ``resultado`` es ``"ok"`` o el tipo de
        429 (``minutely``/``hourly``/... vía :func:`classify_reason`),
        ``http_XXX``, ``transport`` o ``concurrent``.
    """
    try:
        response = session.post(url, data=params, timeout=60)
    except requests.RequestException as exc:
        return "transport", str(exc)
    if response.status_code == 200:
        return "ok", ""
    try:
        body = response.json()
        reason = str(body.get("reason", body))
    except ValueError:
        reason = response.text[:300]
    kind = classify_reason(reason)
    if kind is not None:
        return kind.value, reason
    return f"http_{response.status_code}", reason


def probe_round(
    session: requests.Session,
    url: str,
    params: dict,
    max_attempts: int,
) -> tuple[int, str, str]:
    """Dispara la misma petición hasta 429 o ``max_attempts`` éxitos.

    ``concurrent`` no consume éxitos: espera 5 s y reintenta (máx. 3 veces)
    antes de devolverlo.

    Returns
    -------
    tuple[int, str, str]
        ``(peticiones aceptadas, resultado final, motivo)``. Un resultado
        final ``cap`` significa que no hubo 429 (acotación superior).
    """
    accepted = 0
    concurrent = 0
    while accepted < max_attempts:
        result, reason = fire(session, url, params)
        if result == "ok":
            accepted += 1
            concurrent = 0
            continue
        if result == "concurrent" and concurrent < 3:
            concurrent += 1
            time.sleep(5)
            continue
        return accepted, result, reason
    return accepted, "cap", ""


def wait_for_fresh_minute(margin: float = 2.0) -> None:
    """Duerme hasta el comienzo del siguiente minuto UTC + ``margin``."""
    now = datetime.now(timezone.utc)
    boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
    wait = (boundary - now).total_seconds() + margin
    print(f"[sonda] esperando {wait:.1f}s al comienzo de minuto UTC...")
    time.sleep(wait)


def run_probe(batch_rows: int, max_attempts: int, max_rounds: int) -> int:
    """Ejecuta las rondas del sondeo e imprime el informe. Devuelve exit code."""
    rows = load_merged()
    if len(rows) == 0:
        print("El merged está vacío: no hay filas que sondear.", file=sys.stderr)
        return 1
    size = min(batch_rows, len(rows))
    session = requests.Session()
    rounds: list[tuple[int, int, tuple[float, float]]] = []

    for round_no in range(1, max_rounds + 1):
        wait_for_fresh_minute()
        url, params = build_batch_request(rows.head(size))
        # Misma forma que Client._request: format=flatbuffers en el POST
        params = {**params, "format": "flatbuffers"}
        accepted, result, reason = probe_round(session, url, params, max_attempts)

        # Ventana de minuto desalineada (429 en la 1ª petición): una ronda
        # de reintentos con más margen antes de rendirse.
        if accepted == 0 and result == "minutely":
            print("[sonda] 429 en la primera petición: reintentando ronda")
            wait_for_fresh_minute(margin=5.0)
            accepted, result, reason = probe_round(session, url, params, max_attempts)

        if result in _ABORT_RESULTS:
            print(
                f"[sonda] límite '{result}' antes de poder medir: {reason}\n"
                "Reintenta más tarde (el diario se reinicia a medianoche UTC).",
                file=sys.stderr,
            )
            return 1
        if result.startswith("http_") or result == "transport":
            print(
                f"[sonda] error no relacionado con límites: {reason}", file=sys.stderr
            )
            return 1
        if accepted == 0:
            print(
                f"[sonda] 429 persistente en la primera petición: {reason}",
                file=sys.stderr,
            )
            return 1

        # Rango del peso de ESTE cuerpo, escalado a un lote completo.
        if result == "minutely":
            rng = estimate_weight_range(accepted, _WINDOW_LIMITS[RateLimit.MINUTELY])
        else:  # cap: no hubo 429 -> solo cota superior
            high = (
                _WINDOW_LIMITS[RateLimit.MINUTELY] / (accepted - 1)
                if accepted > 1
                else float("inf")
            )
            rng = (0.0, high)
        assert rng is not None
        scale = batch_rows / size
        scaled = (rng[0] * scale, rng[1] * scale)
        rounds.append((size, accepted, scaled))

        tail = "sin 429 (cota superior)" if result == "cap" else f"429 {result}"
        print(
            f"[sonda] ronda {round_no}: {size} filas, {accepted} aceptadas, {tail}"
            f" -> lote en [{scaled[0]:.1f}, {scaled[1]:.1f})"
        )
        if result == "minutely" and accepted >= 4:
            break
        if size == 1:
            break
        size = max(1, size // 4)

    final = intersect_weight_ranges([r[2] for r in rounds])
    if final is None:
        print(
            "[sonda] AVISO: los rangos de las rondas no se solapan (¿peso no "
            "lineal?); se usa el de la última ronda.",
            file=sys.stderr,
        )
        final = rounds[-1][2]
    low, high = final
    mid = low if high == float("inf") else (low + high) / 2

    print("\n=== Resultado del sondeo ===")
    print(
        f"Peso estimado por lote ({batch_rows} filas): "
        f"[{low:.1f}, {'inf' if high == float('inf') else f'{high:.1f}'})"
        f" ponderadas (~{mid / batch_rows:.2f} por fila)"
    )
    for label, weight in (("bajo", low), ("estimado", mid)):
        advice = pacing_advice(weight)
        print(
            f"Con peso {label}={weight:.1f}: "
            f"{advice['per_minute']} lote(s)/min, {advice['per_hour']}/hora, "
            f"{advice['per_day']}/día, "
            f"intervalo sugerido {advice['interval_seconds']:.0f}s"
        )
    print(
        "Aplica el intervalo del peso ESTIMADO en "
        "configs/project.yaml -> openmeteo.batch_interval_seconds."
    )
    print(
        f"Coste de la sonda: {sum(r[1] for r in rounds)} peticiones aceptadas "
        f"~{sum(a * mid * s / batch_rows for s, a, _ in rounds):.0f} "
        f"ponderadas (~"
        f"{100 * sum(a * mid * s / batch_rows for s, a, _ in rounds) / 10000:.0f} % "
        "del diario)"
    )
    return 0


def main() -> None:
    config = load_config()["openmeteo"]
    parser = argparse.ArgumentParser(description="Mide el peso real de un lote.")
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=config["batch_rows"],
        help="Tamaño del lote a sondear (default: openmeteo.batch_rows)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=15,
        help="Éxitos máximos por ronda antes de parar (default: 15)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Rondas máximas con cuerpos decrecientes (default: 4)",
    )
    args = parser.parse_args()
    sys.exit(run_probe(args.batch_rows, args.max_attempts, args.rounds))


if __name__ == "__main__":
    main()
