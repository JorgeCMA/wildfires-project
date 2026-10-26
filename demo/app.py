"""Wildfire occurrence — Streamlit + Folium demo (PLAN §Plan D).

Two halves:

- **Explorar** — raw FIRMS detections over time (date/region filters), no model.
- **Plan D** — the supervised model's deployment units: A (relative risk bands),
  B (empirical CCAA-day base rate) and C (wide-margin exceedance calibration,
  plus a **live** point scorer: coordinates + date + time → Open-Meteo → the
  persisted occurrence bundle).

Everything is read from the artifacts notebook 01/04 already wrote; the
background pool was frozen on 2026-10-25 (97,630 negatives, ratio ≥ 2), so the
model-derived numbers are final for that snapshot.

Run from the repository root (the theme in ``.streamlit/config.toml`` is read
from the cwd)::

    pip install -e ".[demo]"
    python scripts/fit_occurrence_model.py   # once, for the live Plan-C scorer
    streamlit run demo/app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import folium
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import FastMarkerCluster, HeatMap
from streamlit_folium import st_folium

from wildfire.analysis.demo_data import (
    PALETTE,
    ccaa_base_rate_geojson,
    load_ccaa_rates,
    load_detections,
    load_fold_metrics,
    load_reliability,
    load_scores,
)
from wildfire.analysis.inference import build_point_features
from wildfire.analysis.model_bundle import load_bundle, score_point
from wildfire.analysis.status import format_pool_status, pool_status
from wildfire.config import load_config

SPAIN_CENTER = [40.2, -3.7]
SPAIN_ZOOM = 6

#: Punto del scorer en vivo (Plan C): zoom inicial y límites editables/clic.
POINT_ZOOM = 6
LAT_BOUNDS = (27.0, 44.5)
LON_BOUNDS = (-18.5, 4.5)

#: Logo institucional (un solo asset, sin favicon). Se resuelve relativo a este
#: archivo para que funcione desde cualquier cwd.
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "wildfires_logo_1.png"

#: Basemap común a los tres mapas (cambiar a "CartoDB positron" si los POI de
#: OSM ensucian la lectura).
BASEMAP = "OpenStreetMap"

#: Gradiente secuencial del mapa de calor (densidad 0 → máxima).
HEATMAP_GRADIENT = {
    0.0: PALETTE["bg"],
    0.3: PALETTE["accent"],
    0.55: PALETTE["gold"],
    0.75: PALETTE["primary"],
    1.0: PALETTE["deep"],
}

#: Curvas de fiabilidad: 5 → 20 km, de claro a oscuro.
RELIABILITY_COLORS = {
    5.0: PALETTE["gold"],
    10.0: PALETTE["primary"],
    20.0: PALETTE["deep"],
}

#: Etiquetas visibles de los protocolos. Los valores internos siguen siendo
#: ``season`` / ``ccaa`` (nombres de columnas y artefactos).
PROTOCOL_LABELS = {"season": "Temporada", "ccaa": "Comunidad autónoma"}

#: Ajustes de chrome que el tema de Streamlit no cubre (logo, tabs, CTA). El
#: tamaño base de texto vive en ``.streamlit/config.toml`` (``baseFontSize``).
THEME_CSS = f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {{
    font-size: 1rem;
}}
[data-testid="stImage"] {{
    width: 100%;
    display: flex;
    justify-content: center;
}}
[data-testid="stImage"] img {{
    display: block;
    margin: 0 auto;
    max-width: 100%;
    height: auto;
}}
.st-key-hero {{
    text-align: center;
}}
.st-key-hero [data-testid="stMarkdownContainer"] {{
    text-align: center;
}}
.st-key-hero [data-testid="stCaptionContainer"],
.st-key-hero [data-testid="stCaptionContainer"] p {{
    font-size: 1.1rem;
    text-align: center;
}}
.stTabs [role="tablist"] {{
    width: 100%;
    gap: 0.35rem;
}}
.stTabs [data-testid="stTab"] {{
    flex: 1 1 0;
    min-width: 0;
    justify-content: center;
    height: auto;
    min-height: 3rem;
    padding: 0.7rem 0.4rem;
    font-size: 1.32rem !important;
    font-weight: 600 !important;
    white-space: nowrap;
}}
.stTabs [data-testid="stTab"] [data-testid="stMarkdownContainer"],
.stTabs [data-testid="stTab"] p {{
    overflow: hidden;
    margin: 0;
    font-size: 1.32rem !important;
    font-weight: 600 !important;
    line-height: 1.2;
    text-align: center;
    text-overflow: ellipsis;
}}
.stTabs [data-testid="stTab"][data-selected],
.stTabs [data-testid="stTab"][aria-selected="true"],
.stTabs [data-testid="stTab"][data-selected] [data-testid="stMarkdownContainer"],
.stTabs [data-testid="stTab"][data-selected] p,
.stTabs [data-testid="stTab"][aria-selected="true"] [data-testid="stMarkdownContainer"],
.stTabs [data-testid="stTab"][aria-selected="true"] p {{
    color: {PALETTE["link"]} !important;
}}
a {{color: {PALETTE["link"]} !important;}}
button[kind="primary"],
button[kind="primaryFormSubmit"],
button[data-testid^="stBaseButton-primary"] {{
    background-color: {PALETTE["accent"]} !important;
    border-color: {PALETTE["accent"]} !important;
    color: #000000 !important;
}}
button[kind="primary"]:hover,
button[kind="primaryFormSubmit"]:hover,
button[data-testid^="stBaseButton-primary"]:hover {{
    background-color: {PALETTE["primary"]} !important;
    border-color: {PALETTE["primary"]} !important;
    color: #000000 !important;
}}
</style>
"""

#: Marcador del explorador en modo clúster: llama de fuego en lugar del
#: glyphicon "home" que `FastMarkerCluster` usa por defecto (icon: 'home').
#: Folium envuelve el callback como `var callback = {callback};`, así que aquí
#: va SOLO la expresión función (si se incluye `var callback = ...` el script
#: resultante es un SyntaxError y el mapa no se dibuja).
FIRE_MARKER_CALLBACK = """
    function (row) {
        return L.marker(L.latLng(row[0], row[1]), {
            icon: L.AwesomeMarkers.icon({
                prefix: 'fa', icon: 'fire', markerColor: 'red',
            }),
        });
    }
"""

st.set_page_config(
    page_title="Riesgo de incendio — priorización operativa",
    layout="wide",
)
st.markdown(THEME_CSS, unsafe_allow_html=True)


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


@st.cache_resource(show_spinner="Cargando el modelo de ocurrencia…")
def _bundle():
    return load_bundle(load_config()["output"]["model_bundle"])


def _safe(loader, *args, **kwargs):
    """Load an artifact, degrading to a clear error instead of a traceback."""
    try:
        return loader(*args, **kwargs)
    except FileNotFoundError as exc:
        st.error(
            "No se encontró un artefacto requerido. Ejecuta el notebook 01 "
            "(detecciones) o 04 (salidas del modelo) y recarga la página."
        )
        with st.expander("Detalle técnico"):
            st.code(str(exc), language="text")
        return None


def _fmt_int(value: int) -> str:
    """Thousands separator as in Spanish (``97.630``, not ``97,630``)."""
    return f"{int(value):,}".replace(",", ".")


def _tab_intro(text: str) -> None:
    """Short, consistent description at the top of each section tab."""
    st.markdown(
        f"<div style='background:{PALETTE['secondary_bg']}; "
        f"border-left: 4px solid {PALETTE['primary']}; "
        f"padding: 0.7rem 1rem; border-radius: 0.45rem; "
        f"margin: 0.2rem 0 1rem 0; font-size: 1rem;'>"
        f"<strong>{text}</strong></div>",
        unsafe_allow_html=True,
    )


def _frozen_badge(label: str) -> None:
    """Frozen-pool chip, with a caption fallback for older Streamlit versions."""
    if hasattr(st, "badge"):
        st.badge(label, icon=":material/lock:", color="orange")
    else:
        st.caption(label)


with st.container(key="hero", horizontal_alignment="center"):
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=420)
    st.title("Riesgo de incendio — priorización operativa")
    st.caption(
        "Ranking de ocurrencia y excedencia a partir de detecciones FIRMS, "
        "clima y cobertura del suelo."
    )
    _st = _pool_status()
    _frozen_badge(
        f"Pool de fondo congelado · {_fmt_int(_st['pool_rows'])} negativos · 25-10-2026"
    )

# --- Sections ---------------------------------------------------------------
tab_explorar, tab_a, tab_b, tab_c, tab_estado = st.tabs(
    [
        "Explorar detecciones",
        "Tramos de riesgo",
        "Tasa base por CCAA",
        "Excedencia calibrada",
        "Estado del modelo",
    ]
)

# 1) Raw detections over time ----------------------------------------------
with tab_explorar:
    st.subheader("Dónde ardieron: detecciones FIRMS en el tiempo")
    _tab_intro(
        "Detecciones crudas de FIRMS en el mapa y su evolución mensual. Filtra "
        "por fecha y comunidad, y alterna entre clúster y mapa de calor; esta "
        "pestaña no usa el modelo."
    )
    detections = _safe(_detections)
    if detections is None or detections.empty:
        st.info("Sin detecciones para mostrar.")
    else:
        dmin = detections["acq_date"].min().date()
        dmax = detections["acq_date"].max().date()
        col_dates, col_ccaa = st.columns([1, 1])
        date_range = col_dates.date_input(
            "Rango de fechas",
            value=(dmin, dmax),
            min_value=dmin,
            max_value=dmax,
        )
        ccaa_options = sorted(detections["ccaa"].dropna().unique())
        selected_ccaa = col_ccaa.multiselect(
            "Comunidades autónomas",
            ccaa_options,
            default=[],
            placeholder="Ninguna",
            select_all=False,
            help="Sin selección se muestran todas.",
        )

        view = detections
        if date_range and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            view = view[view["acq_date"].between(start, end)]
        if selected_ccaa:
            view = view[view["ccaa"].isin(selected_ccaa)]

        mode = st.radio("Capa", ["Clúster", "Mapa de calor"], horizontal=True)
        st.caption(f"{_fmt_int(len(view))} detecciones mostradas.")

        fmap = folium.Map(location=SPAIN_CENTER, zoom_start=SPAIN_ZOOM, tiles=BASEMAP)
        points = view[["latitude", "longitude"]].dropna().to_numpy().tolist()
        if mode == "Mapa de calor":
            HeatMap(points, radius=6, blur=8, gradient=HEATMAP_GRADIENT).add_to(fmap)
        else:
            FastMarkerCluster(points, callback=FIRE_MARKER_CALLBACK).add_to(fmap)
        st_folium(fmap, height=580, use_container_width=True, returned_objects=[])

        monthly = (
            view.assign(month=view["acq_date"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
        )
        if not monthly.empty:
            st.bar_chart(monthly.rename("detecciones"), color=PALETTE["primary"])
            st.caption("Detecciones por mes en el filtro actual.")

# 2) Plan A — relative risk bands ------------------------------------------
with tab_a:
    st.subheader("Tramos de riesgo relativo")
    _tab_intro(
        "Puntuación media de ocurrencia por comunidad para el protocolo elegido, "
        "ordenada de mayor a menor. Los tramos son quintiles relativos: ordenan "
        "el lote, no expresan una probabilidad absoluta."
    )
    protocol = st.radio(
        "Protocolo de validación",
        ["season", "ccaa"],
        format_func=lambda value: PROTOCOL_LABELS[value],
        horizontal=True,
        help=(
            "Temporada bloquea por trimestre-año; Comunidad autónoma, por "
            "comunidad. Cambia el lote puntuado."
        ),
    )
    scores = _safe(_scores, protocol)
    if scores is not None:
        by_ccaa = (
            scores.groupby("ccaa")["score"]
            .mean()
            .sort_values(ascending=False)
            .rename("Puntuación media")
        )
        st.dataframe(
            by_ccaa.reset_index(),
            width="stretch",
            hide_index=True,
            column_config={
                "ccaa": st.column_config.TextColumn("CCAA", alignment="right"),
                "Puntuación media": st.column_config.NumberColumn(
                    "Puntuación media", alignment="right"
                ),
            },
        )

# 3) Plan B — empirical CCAA-day base rate ---------------------------------
with tab_b:
    st.subheader("Tasa base por CCAA y día")
    _tab_intro(
        "Tasa base observada de incendios por comunidad y día (2023–2024). Es "
        "una frecuencia empírica, sin modelo, para contextualizar el ranking."
    )
    rates = _safe(_rates)
    if rates is not None:
        geojson = _safe(_choropleth_geojson)
        if geojson is not None:
            region_rates = rates[rates["ccaa"] != "ALL"]
            values = region_rates["fire_rate"].astype(float)
            ramp = LinearColormap(
                [
                    "#FFF0CF",
                    PALETTE["accent"],
                    PALETTE["gold"],
                    PALETTE["primary"],
                    PALETTE["deep"],
                ],
                vmin=float(values.min()) * 100,
                vmax=float(values.max()) * 100,
                caption="Tasa base CCAA-día (%)",
            )

            def _pct(value) -> str:
                return (
                    f"{value * 100:.2f} %"
                    if value is not None and pd.notna(value)
                    else "sin datos"
                )

            geo = {
                **geojson,
                "features": [
                    {
                        **feature,
                        "properties": {
                            **feature["properties"],
                            "tasa_pct": _pct(feature["properties"].get("fire_rate")),
                        },
                    }
                    for feature in geojson["features"]
                ],
            }

            def _rate_style(feature: dict) -> dict:
                value = feature["properties"].get("fire_rate")
                color = (
                    ramp(value * 100)
                    if value is not None and pd.notna(value)
                    else PALETTE["grid"]
                )
                return {
                    "fillColor": color,
                    "color": PALETTE["muted"],
                    "weight": 0.6,
                    "fillOpacity": 0.75,
                    "opacity": 0.45,
                }

            fmap = folium.Map(
                location=SPAIN_CENTER, zoom_start=SPAIN_ZOOM, tiles=BASEMAP
            )
            folium.GeoJson(
                geo,
                style_function=_rate_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=["ccaa_es", "tasa_pct"],
                    aliases=["CCAA", "Tasa base"],
                    localize=True,
                ),
                name="Tasa base",
            ).add_to(fmap)
            ramp.add_to(fmap)
            st_folium(fmap, height=520, use_container_width=True, returned_objects=[])

        st.dataframe(
            rates,
            width="stretch",
            hide_index=True,
            column_config={
                "ccaa": st.column_config.TextColumn("Comunidad autónoma"),
                "n_ccaa_days": st.column_config.NumberColumn(
                    "Rango de días", format="%d"
                ),
                "n_fire_days": st.column_config.NumberColumn(
                    "Días con detecciones", format="%d"
                ),
                "fire_rate": st.column_config.NumberColumn(
                    "Ratio de fuego", format="percent"
                ),
            },
        )

# 4) Plan C — exceedance calibration (curves + live point scorer) -----------
with tab_c:
    st.subheader("Excedencia calibrada")
    _tab_intro(
        "Puntúa un punto y una fecha con el scorer y, al final, revisa la "
        "fiabilidad de la probabilidad de excedencia a 5/10/20 km. La "
        "probabilidad es de margen ancho, no puntual."
    )
    st.subheader("Puntuar un punto y una fecha")
    st.caption(
        "Haz clic en el mapa o escribe coordenadas, fecha y hora: se descarga "
        "el clima de Open-Meteo, se rellenan CLC/CCAA/presupuestos y se puntúa "
        "con el bundle persistido."
    )

    # Punto activo: persiste entre reruns y lo mueve el clic en el mapa.
    if "pc_lat" not in st.session_state:
        st.session_state.pc_lat = 40.4168
    if "pc_lon" not in st.session_state:
        st.session_state.pc_lon = -3.7038

    picker = folium.Map(
        location=[st.session_state.pc_lat, st.session_state.pc_lon],
        zoom_start=POINT_ZOOM,
        tiles=BASEMAP,
    )
    folium.Marker(
        [st.session_state.pc_lat, st.session_state.pc_lon],
        tooltip=f"{st.session_state.pc_lat:.4f}, {st.session_state.pc_lon:.4f}",
    ).add_to(picker)
    clicked = st_folium(
        picker,
        height=580,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key="pc_picker",
    )
    hit = (clicked or {}).get("last_clicked")
    if hit:
        clicked_at = (round(float(hit["lat"]), 4), round(float(hit["lng"]), 4))
        # El componente devuelve el mismo clic en cada rerun: aplícalo una vez.
        if clicked_at != st.session_state.get("pc_last_click"):
            st.session_state.pc_last_click = clicked_at
            st.session_state.pc_lat = min(
                max(clicked_at[0], LAT_BOUNDS[0]), LAT_BOUNDS[1]
            )
            st.session_state.pc_lon = min(
                max(clicked_at[1], LON_BOUNDS[0]), LON_BOUNDS[1]
            )
            st.rerun()

    with st.form("live_point"):
        col1, col2, col3 = st.columns(3)
        lat = col1.number_input(
            "Latitud",
            min_value=LAT_BOUNDS[0],
            max_value=LAT_BOUNDS[1],
            step=0.0001,
            format="%.4f",
            key="pc_lat",
        )
        lon = col2.number_input(
            "Longitud",
            min_value=LON_BOUNDS[0],
            max_value=LON_BOUNDS[1],
            step=0.0001,
            format="%.4f",
            key="pc_lon",
        )
        day = col3.date_input(
            "Fecha",
            value=pd.Timestamp("2024-07-15").date(),
            min_value=pd.Timestamp("2020-01-01").date(),
            max_value=pd.Timestamp("2025-12-31").date(),
        )
        col4, col5 = st.columns(2)
        hour = col4.slider("Hora (UTC)", 0, 23, 12)
        margin = col5.selectbox("Margen", [5.0, 10.0, 20.0], index=1)
        submitted = st.form_submit_button("Calcular excedencia")

    if submitted:
        try:
            bundle = _bundle()
        except FileNotFoundError as exc:
            st.error(
                "Falta el bundle del modelo. Ejecuta "
                "`python scripts/fit_occurrence_model.py` y recarga la página."
            )
            with st.expander("Detalle técnico"):
                st.code(str(exc), language="text")
        else:
            try:
                with st.spinner("Descargando clima y calculando features…"):
                    features = build_point_features(lat, lon, day, hour)
                    result = score_point(bundle, features)
            except Exception as exc:  # noqa: BLE001 — surface any failure to the user
                st.error(
                    "No se pudo puntuar el punto. Revisa las coordenadas y la "
                    "fecha, o inténtalo de nuevo."
                )
                with st.expander("Detalle técnico"):
                    st.code(str(exc), language="text")
            else:
                exceedance = result["exceedance"]
                prob = exceedance[float(margin)]
                st.metric(
                    f"P(≥1 detección a ≤ {margin:g} km ese día)",
                    f"{prob * 100:.3f} %",
                )
                st.caption(f"Score bruto del modelo: {result['score']:.4f}")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "margen_km": list(exceedance),
                            "probabilidad": [
                                round(value * 100, 4) for value in exceedance.values()
                            ],
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "margen_km": st.column_config.NumberColumn(
                            "Kilómetros a la redonda", format="%g"
                        ),
                        "probabilidad": st.column_config.NumberColumn("Probabilidad"),
                    },
                )

    st.divider()
    st.subheader("Fiabilidad por margen")
    st.caption(
        "Probabilidad predicha frente a frecuencia observada en 5, 10 y 20 km; "
        "la diagonal punteada es el ideal. Calibración isotónica ajustada en 2023 "
        "y calibrada en 2024, con corrección de prior."
    )
    rel = _safe(_reliability)
    if rel is not None:
        col_plot, col_table = st.columns(2, gap="large")
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        fig.patch.set_facecolor(PALETTE["bg"])
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=1,
            color=PALETTE["muted"],
            label="ideal",
        )
        for margin, chunk in rel.groupby("margin_km"):
            chunk = chunk.sort_values("mean_predicted")
            ax.plot(
                chunk["mean_predicted"],
                chunk["observed_freq"],
                marker="o",
                markersize=4,
                color=RELIABILITY_COLORS.get(float(margin), PALETTE["primary"]),
                label=f"{margin:g} km",
            )
        ax.set_xlabel("Probabilidad predicha", color=PALETTE["muted"], fontsize=10)
        ax.set_ylabel("Frecuencia observada", color=PALETTE["muted"], fontsize=10)
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(PALETTE["grid"])
        ax.tick_params(colors=PALETTE["muted"], labelsize=10)
        ax.set_axisbelow(True)
        ax.grid(color=PALETTE["grid"], linewidth=0.8)
        ax.legend(title="margen", frameon=False, fontsize=10, title_fontsize=10)
        with col_plot, st.container(horizontal_alignment="center"):
            st.pyplot(fig, width=480)
        plt.close(fig)
        with col_table:
            st.dataframe(rel, width="stretch", hide_index=True, height=320)

# 5) Status -----------------------------------------------------------------
with tab_estado:
    st.subheader("Estado del modelo y notas de uso")
    _tab_intro(
        "Estado del pool de negativos, métricas del modelo y limitaciones que "
        "conviene tener presentes al interpretar los resultados."
    )
    st.code(format_pool_status(_pool_status()), language="text")
    st.markdown(
        "- El **mapa** usa solo detecciones crudas: no es el modelo.\n"
        "- Los **tramos de riesgo** son **relativos** y la **excedencia "
        "calibrada** es de **margen ancho**, no una probabilidad puntual.\n"
        "- El PR-AUC está inflado por el enmarcado ~1:2: es un ranking, no "
        "precisión de campo.\n"
        "- Detalle del pool: `docs/background-pool-status.md`."
    )
    fold = _safe(_fold_metrics)
    if fold is not None:
        st.dataframe(fold, width="stretch", hide_index=True)

st.divider()
st.caption(
    "Fuentes: NASA FIRMS · CLCPlus · Open-Meteo — datos 2023–2024 · "
    "actualizado 25-10-2026."
)
