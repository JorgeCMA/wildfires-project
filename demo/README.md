# Wildfire demo (Streamlit + Folium)

Read-only viewer for the supervised occurrence model's deployment units
(PLAN §Plan D), plus a **live** Plan-C point scorer. The browse sections read
the artifacts notebooks 01/04 already wrote; only the live scorer spends Open-Meteo
quota (one request per point you submit).

## Install & run

From the repository root:

```powershell
pip install -e ".[demo]"
python scripts/fit_occurrence_model.py   # once: writes the Plan-C bundle
streamlit run demo/app.py
```

The app anchors data paths to `PROJECT_ROOT` and the logo to the app file, so
it can be launched from any working directory; launch from the repo root so
`.streamlit/config.toml` (the theme) also applies. The browse sections need these
artifacts (regenerate by running the matching notebook):

| Artifact | Notebook |
|---|---|
| `data/processed/expanded/firms_spain_final_lags_index.csv` | 05 / pipeline |
| `data/processed/predictions/occurrence_scores.csv` | 04 |
| `data/processed/predictions/ccaa_day_fire_rates.csv` | 04 |
| `data/processed/predictions/calibration_reliability.csv` | 04 |
| `data/processed/predictions/occurrence_fold_metrics.csv` | 04 |
| `data/raw/ccaa/spain-communities.geojson` | team-provided |

The **live Plan-C scorer** additionally needs the persisted bundle
`data/processed/models/occurrence_bundle.joblib` (versioned and reproducible),
written by `scripts/fit_occurrence_model.py` (see `docs/calibration-C.md`).

If one is missing the app shows which artifact is absent instead of crashing.

## Theme and assets

The demo uses the corporate palette, defined once in
`src/wildfire/analysis/demo_data.py` (`PALETTE`) and mirrored in
`.streamlit/config.toml` for Streamlit's own chrome (light mode: primary
`#F65C2C`, background `#FFF9EA`, secondary `#FFF6E5`, black text). Charts and
Folium layers read the same tokens; `RISK_BAND_COLORS` is the low→high ramp.

- **Logo**: `demo/assets/wildfires_logo_1.png` (transparent PNG), centered at
  the top of the main area with `st.image` (420 px wide).
- The header (logo/title/description/badge) is centered; the five tabs split the
  **full content width** evenly (side margins preserved), with larger type
  (root 18 px, tabs 1.32 rem).
- Streamlit reads `.streamlit/config.toml` from the **current working
  directory**: launch from the repo root so the theme applies
  (`streamlit run demo/app.py`).
- Theme/contrast guard: `pytest tests/test_demo_theme.py`.

## Sections (tabs)

Five tabs sit between the static header (logo, title, description, pool badge)
and the section content; there is no sidebar. Each section opens with a short
bold description and holds its own controls at the top of its tab.

1. **Explorar detecciones** — raw FIRMS detections over time (date-range + CCAA
   filters; cluster / heatmap toggle). No model.
2. **Tramos de riesgo** — relative risk-band ranking for a chosen protocol, with
   the mean score by CCAA (the equal-quintile distribution chart was removed).
3. **Tasa base por CCAA** — empirical `P(≥1 detection | CCAA, day)` choropleth.
4. **Excedencia calibrada** — the live scorer (clickable map, date/hour) **and**,
   at the end, the reliability curves plus their table side by side (50/50).
5. **Estado del modelo** — background-pool status + usage caveats.

Guía completa en español de cada sección y control: `docs/guia-demo.md`.

## Background pool

The negatives pool is **complete and frozen** (2026-10-25): 97,630 usable rows,
every year above the configured `background_ratio: 2` (2023: 3.31×, 2024:
5.80×). Model-derived numbers are final for that snapshot. Check the live number
with `python scripts/background_pool_status.py`; details in
`docs/background-pool-status.md`.

Caveats that are not about pool size: the model is a **ranking** (not a point
probability), the route-C exceedance probability is a wide-margin approximation
(5/10/20 km), and PR-AUC is inflated by the ~1:2 framing.
