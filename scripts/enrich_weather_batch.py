"""Descarga por lotes el clima de Open-Meteo reintentando según el límite.

Ejecuta ``run_weather_enrichment`` sin supervisión: baja lote a lote
(``openmeteo.batch_rows`` filas) hasta completar el CSV de reanudación,
esperando lo que toque si la API devuelve 429 (minuto -> espera al siguiente
minuto; hora -> a la siguiente hora; día -> parada limpia, re-ejecutar para
continuar). Si termina completo, publica además el CSV de clima terminado
(``finalize_weather_csv``), que es la entrada de la celda de clima del
notebook 01. Código de salida: 0 completado o ``--max-batches`` alcanzado,
3 parada por límite diario, 4 parada por esperas consecutivas excesivas,
1 error no relacionado con límites (traceback).

Uso::

    python scripts/enrich_weather_batch.py
    python scripts/enrich_weather_batch.py --max-batches 5 --interval 90
"""

from __future__ import annotations

import argparse
import sys

from wildfire.enrichment.weather_batch import (
    WeatherBatchStatus,
    finalize_weather_csv,
    run_weather_enrichment,
)

# 3 y 4 documentados arriba: permiten distinguir "reanudar mañana" de
# "algo va mal con los reintentos" en un script que llame a este proceso.
_EXIT_CODES = {
    WeatherBatchStatus.COMPLETE: 0,
    WeatherBatchStatus.MAX_BATCHES: 0,
    WeatherBatchStatus.DAILY_STOP: 3,
    WeatherBatchStatus.WAIT_CAP: 4,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich FIRMS data with Open-Meteo batch weather (resumable)."
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after N batches in this run (default: run to completion)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between successful batches "
        "(default: openmeteo.batch_interval_seconds from configs/project.yaml)",
    )
    args = parser.parse_args()

    status = run_weather_enrichment(
        max_batches=args.max_batches,
        interval_seconds=args.interval,
    )
    print(f"Status: {status.value}")

    # Descarga completa -> publicar el CSV de clima terminado (entrada de la
    # celda de clima del notebook 01). Idempotente: si ya existía, no copia.
    if status is WeatherBatchStatus.COMPLETE:
        completed = finalize_weather_csv()
        print(f"Weather CSV: {completed}")

    sys.exit(_EXIT_CODES[status])


if __name__ == "__main__":
    main()
