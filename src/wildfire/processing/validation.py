"""Data quality validation for FIRMS and enriched datasets.

Todas las funciones devuelven avisos y nunca lanzan: la validación es una
puerta que informa, no que bloquea. La contrapartida es que las columnas
*esperadas pero ausentes* también avisan (modo fail-closed): un CSV a medio
enriquecer no puede salir "limpio" por no tener las columnas.
"""

from __future__ import annotations

import pandas as pd

from wildfire.config import load_config


def _expected_weather_fields() -> list[str]:
    """Las 15 variables horarias de producción (config, no hardcodeado)."""
    return list(load_config()["openmeteo"]["batch_query"]["hourly"])


def _expected_clc_fields() -> list[str]:
    """``clc_class`` + vecinos según los anillos activos en config."""
    # Importación diferida: `merge_sensors` importa este módulo, así que
    # importar `enrichment` arriba crearía un ciclo (ver commit que lo rompió).
    from wildfire.enrichment.clc_enrichment import NEIGHBORHOOD_RINGS

    config = load_config().get("clcplus", {})
    neighborhood = config.get("neighborhood", {})
    rings = neighborhood.get("rings", []) if neighborhood.get("enabled") else []
    fields = ["clc_class"]
    for ring in rings:
        fields.extend(f"clc_class_{s}" for _, _, s in NEIGHBORHOOD_RINGS.get(ring, []))
    return fields


def validate_firms(df: pd.DataFrame) -> list[str]:
    """Run basic quality checks on a FIRMS DataFrame.

    Returns a list of warning messages. An empty list means all checks passed.

    Parameters
    ----------
    df:
        FIRMS DataFrame (merged or raw).

    Returns
    -------
    list[str]
        Validation warnings.
    """
    warnings: list[str] = []

    required = ["latitude", "longitude", "acq_date", "acq_time", "confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.append(f"Missing required columns: {missing}")

    if "latitude" in df.columns:
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        if lat.isna().any():
            warnings.append(f"{int(lat.isna().sum())} rows have missing latitude")
        out_of_range = ~lat.between(-90, 90) & lat.notna()
        if out_of_range.any():
            n = int(out_of_range.sum())
            warnings.append(f"{n} rows have latitude outside [-90, 90]")

    if "longitude" in df.columns:
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        if lon.isna().any():
            warnings.append(f"{int(lon.isna().sum())} rows have missing longitude")
        out_of_range = ~lon.between(-180, 180) & lon.notna()
        if out_of_range.any():
            n = int(out_of_range.sum())
            warnings.append(f"{n} rows have longitude outside [-180, 180]")

    if "frp" in df.columns:
        frp = pd.to_numeric(df["frp"], errors="coerce")
        neg_frp = int(((frp < 0) & frp.notna()).sum())
        if neg_frp > 0:
            warnings.append(f"{neg_frp} rows have negative FRP values")

    if "acq_date" in df.columns:
        parsed = pd.to_datetime(df["acq_date"], format="%Y-%m-%d", errors="coerce")
        bad = int(parsed.isna().sum())
        if bad:
            warnings.append(f"{bad} rows have unparseable acq_date (expected YYYY-MM-DD)")

    if df.empty:
        warnings.append("DataFrame is empty")

    return warnings


def _missing_or_gappy(df: pd.DataFrame, cols: list[str], gap: str) -> list[str]:
    """Aviso por columna esperada ausente + aviso por huecos en presentes."""
    warnings: list[str] = []
    absent = [c for c in cols if c not in df.columns]
    if absent:
        warnings.append(f"Missing expected columns ({gap}): {absent}")
    for col in cols:
        if col in df.columns and df[col].isna().any():
            n = int(df[col].isna().sum())
            warnings.append(f"{n} rows have missing {col} values ({gap})")
    return warnings


def validate_enriched_clc(df: pd.DataFrame) -> list[str]:
    """Run quality checks for CLCPlus land cover enrichment.

    Checks that ``clc_class``, the neighbor columns of the active rings
    and ``clc_uniform_surroundings`` exist and are populated.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.

    Returns
    -------
    list[str]
        Validation warnings.
    """
    cols = _expected_clc_fields() + ["clc_uniform_surroundings"]
    return _missing_or_gappy(df, cols, "CLC enrichment gap")


def validate_enriched_openmeteo(df: pd.DataFrame) -> list[str]:
    """Run quality checks for Open-Meteo weather enrichment.

    Checks all 15 production variables (config, not a hardcoded subset).

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.

    Returns
    -------
    list[str]
        Validation warnings.
    """
    return _missing_or_gappy(df, _expected_weather_fields(), "weather enrichment gap")


def validate_enriched_ccaa(df: pd.DataFrame) -> list[str]:
    """Run quality checks for the CCAA region + budget step.

    Checks that ``ccaa`` exists and that the rolling budget sums exist;
    missing sums are legitimate for regions without budget (Ceuta/Melilla),
    so they only inform with a count.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.

    Returns
    -------
    list[str]
        Validation warnings.
    """
    cols = ["ccaa", "sum_prevention", "sum_extinction"]
    return _missing_or_gappy(df, cols, "CCAA/budget gap")


def validate_enriched(df: pd.DataFrame) -> list[str]:
    """Run all quality checks on an enriched FIRMS DataFrame.

    Calls ``validate_firms``, ``validate_enriched_clc``,
    ``validate_enriched_openmeteo`` and ``validate_enriched_ccaa``,
    returning the combined warnings.

    Parameters
    ----------
    df:
        Enriched FIRMS DataFrame.

    Returns
    -------
    list[str]
        Validation warnings.
    """
    warnings = validate_firms(df)
    warnings.extend(validate_enriched_clc(df))
    warnings.extend(validate_enriched_openmeteo(df))
    warnings.extend(validate_enriched_ccaa(df))
    return warnings
