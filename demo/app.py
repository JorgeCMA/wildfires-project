"""Wildfire occurrence — Streamlit + Folium demo (PLAN §Plan D, provisional).

Two halves:

- **Explorar** — raw FIRMS detections over time (date/region filters), no model.
- **Plan D** — the supervised model's deployment units: A (relative risk bands),
  B (empirical CCAA-day base rate) and C (wide-margin exceedance calibration).

Everything is read from the artifacts notebook 01/04 already wrote; the pool is
still being filled, so every model-derived number is **provisional**.

Run from the repository root::

    pip install -e ".[demo]"
    streamlit run demo/app.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from folium.plugins import FastMarkerCluster, HeatMap
from streamlit_folium import st_folium

from wildfire.analysis.demo_data import (
    RISK_BAND_COLORS,
    band_counts,
    ccaa_base_rate_geojson,
    load_ccaa_rates,
    load_detections,
    load_fold_metrics,
    load_reliability,
    load_scores,
)
from wildfire.analysis.status import format_pool_status, pool_status

SPAIN_CENTER = [40.2, -3.7]
SPAIN_ZOOM = 6

st.set_page_config(page_title="Wildfire risk — demo", layout="wide")


@st.cache_data(show_spinner="Cargando detecciones…")
def _detections() -> pd.DataFrame:
    return load_detections()


@st.cache_data(show_spinner="Cargando scores…")
def _scores(protocol: str) -> pd.DataFrame:
    return load_scores(protocol)


@st.cache_data(show_spinner=False)
def _rates() -> pd.DataFrame:
    return load_ccaa_rates()


@st.cache_data(show_spinner=False)
def _reliability() -> pd.DataFrame:
    return load_reliability()


@st.cache_data(show_spinner=False)
def _fold_metrics() -> pd.DataFrame:
    return load_fold_metrics()


@st.cache_data(show_spinner="Preparando el mapa…")
def _choropleth_geojson() -> dict:
    return ccaa_base_rate_geojson(_rates())


@st.cache_resource(show_spinner=False)
def _pool_status() -> dict:
    return pool_status()


def _safe(loader, *args, **kwargs):
    """Load an artifact, degrading to a clear error instead of a traceback."""
    try:
        return loader(*args, **kwargs)
    except FileNotFoundError as exc:
        st.error(
            f"Falta un artefacto: `{exc}`. Ejecuta el notebook que lo genera "
            "(01 para detecciones, 04 para las salidas del modelo)."
        )
        return None


def _band_bar(counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = [RISK_BAND_COLORS.get(band, "#999999") for band in counts.index]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors)
    ax.set_ylabel("Registros")
    ax.set_title("Distribución por tramo (relativo)")
    st.pyplot(fig)
    plt.close(fig)


st.title("Wildfire occurrence — demo (provisional)")
st.warning(
    "**Demo provisional.** El fondo (negativos) aún se está rellenando "
    f"({_pool_status()['pool_rows']:,} filas): todas las cifras derivadas del "
    "modelo pueden cambiar. Ver la pestaña *Estado*."
)

# --- Sidebar ---------------------------------------------------------------
detections = _safe(_detections)
with st.sidebar:
    st.header("Controles")
    protocol = st.radio("Protocolo de validación", ["season", "ccaa"], horizontal=True)
    st.divider()
    st.subheader("Mapa (detecciones)")
    date_range = None
    ccaa_options: list[str] = []
    if detections is not None and not detections.empty:
        dmin = detections["acq_date"].min().date()
        dmax = detections["acq_date"].max().date()
        date_range = st.date_input(
            "Rango de fechas", value=(dmin, dmax), min_value=dmin, max_value=dmax
        )
        ccaa_options = sorted(detections["ccaa"].dropna().unique())
    selected_ccaa = st.multiselect("Comunidades", ccaa_options, default=[])
    st.divider()
    st.caption(
        "La pestaña **Plan C (live)** con coordenadas + Open-Meteo llega en la "
        "fase 2; aquí C muestra las curvas de fiabilidad ya calculadas."
    )

# --- Tabs ------------------------------------------------------------------
tab_map, tab_a, tab_b, tab_c, tab_status = st.tabs(
    [
        "Explorar mapa",
        "Plan A — tramos",
        "Plan B — tasa base",
        "Plan C — excedencia",
        "Estado",
    ]
)

# 1) Raw detections over time ----------------------------------------------
with tab_map:
    st.subheader("Dónde ardieron: detecciones FIRMS en el tiempo")
    if detections is None or detections.empty:
        st.info("Sin detecciones para mostrar.")
    else:
        view = detections
        if date_range and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            view = view[view["acq_date"].between(start, end)]
        if selected_ccaa:
            view = view[view["ccaa"].isin(selected_ccaa)]

        mode = st.radio("Capa", ["Clúster", "Mapa de calor"], horizontal=True)
        st.caption(f"{len(view):,} detecciones mostradas.")

        fmap = folium.Map(
            location=SPAIN_CENTER, zoom_start=SPAIN_ZOOM, tiles="OpenStreetMap"
        )
        points = view[["latitude", "longitude"]].dropna().to_numpy().tolist()
        if mode == "Mapa de calor":
            HeatMap(points, radius=6, blur=8).add_to(fmap)
        else:
            FastMarkerCluster(points).add_to(fmap)
        st_folium(fmap, height=580, use_container_width=True, returned_objects=[])

        monthly = (
            view.assign(month=view["acq_date"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
        )
        if not monthly.empty:
            st.bar_chart(monthly.rename("detecciones"))

# 2) Plan A — relative risk bands ------------------------------------------
with tab_a:
    st.subheader("Plan A — tramos de riesgo relativos")
    st.caption(
        "Los tramos son relativos al lote puntuado (por protocolo), no una "
        "escala absoluta de peligro. Modelo base: logístico."
    )
    scores = _safe(_scores, protocol)
    if scores is not None:
        counts = band_counts(scores)
        _band_bar(counts)
        by_ccaa = (
            scores.groupby("ccaa")["score"]
            .mean()
            .sort_values(ascending=False)
            .rename("score_medio")
        )
        st.dataframe(by_ccaa.to_frame(), width="stretch")

# 3) Plan B — empirical CCAA-day base rate ---------------------------------
with tab_b:
    st.subheader("Plan B — tasa base empírica CCAA × día")
    st.caption("P(≥1 detección en la CCAA ese día), estimada de las detecciones FIRMS.")
    rates = _safe(_rates)
    if rates is not None:
        geojson = _safe(_choropleth_geojson)
        if geojson is not None:
            region_rates = rates[rates["ccaa"] != "ALL"]
            fmap = folium.Map(
                location=SPAIN_CENTER, zoom_start=SPAIN_ZOOM, tiles="OpenStreetMap"
            )
            folium.Choropleth(
                geo_data=geojson,
                data=region_rates,
                columns=["ccaa", "fire_rate"],
                key_on="feature.properties.ccaa_es",
                fill_color="YlOrRd",
                fill_opacity=0.75,
                line_opacity=0.4,
                legend_name="Tasa base CCAA-día",
                nan_fill_color="#f0f0f0",
            ).add_to(fmap)
            st_folium(fmap, height=520, use_container_width=True, returned_objects=[])

        display = rates.copy()
        display["fire_rate"] = (display["fire_rate"] * 100).round(2)
        st.dataframe(display, width="stretch", hide_index=True)

# 4) Plan C — exceedance calibration (precomputed curves) ------------------
with tab_c:
    st.subheader("Plan C — calibración de excedencia (margen ancho)")
    st.caption(
        "PROVISIONAL. P(≥1 detección a ≤ R km ese día); isotónica + corrección "
        "de prior. La versión *live* (coordenadas + Open-Meteo) llega en la fase 2."
    )
    rel = _safe(_reliability)
    if rel is not None:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        for margin, chunk in rel.groupby("margin_km"):
            chunk = chunk.sort_values("mean_predicted")
            ax.plot(
                chunk["mean_predicted"],
                chunk["observed_freq"],
                marker="o",
                label=f"{margin:g} km",
            )
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Frecuencia observada")
        ax.set_title("Fiabilidad por margen")
        ax.legend(title="margen")
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(rel, width="stretch", hide_index=True)

# 5) Status / provisionality -----------------------------------------------
with tab_status:
    st.subheader("Estado del fondo y notas de provisionalidad")
    st.code(format_pool_status(_pool_status()), language="text")
    st.markdown(
        "- El **mapa** usa solo detecciones crudas: no es el modelo.\n"
        "- Los tramos (A) son **relativos** y la probabilidad de excedencia (C) "
        "es **provisional y de margen ancho**.\n"
        "- El PR-AUC está inflado por el enmarcado ~1:2: es un ranking, no "
        "precisión de campo.\n"
        "- Documento completo: `docs/provisional-status.md`."
    )
    fold = _safe(_fold_metrics)
    if fold is not None:
        st.dataframe(fold, width="stretch", hide_index=True)
