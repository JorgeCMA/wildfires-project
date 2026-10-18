# Wildfire demo (Streamlit + Folium)

Read-only viewer for the supervised occurrence model's deployment units
(PLAN §Plan D). Everything comes from the artifacts notebooks 01/04 already
wrote; the app never spends Open-Meteo quota.

## Install & run

From the repository root:

```powershell
pip install -e ".[demo]"
streamlit run demo/app.py
```

The app anchors all paths to `PROJECT_ROOT`, so it can be launched from any
working directory. It needs these artifacts (regenerate by running the matching
notebook):

| Artifact | Notebook |
|---|---|
| `data/processed/expanded/firms_spain_final_lags_index.csv` | 05 / pipeline |
| `data/processed/predictions/occurrence_scores.csv` | 04 |
| `data/processed/predictions/ccaa_day_fire_rates.csv` | 04 |
| `data/processed/predictions/calibration_reliability.csv` | 04 |
| `data/processed/predictions/occurrence_fold_metrics.csv` | 04 |
| `data/raw/ccaa/spain-communities.geojson` | team-provided |

If one is missing the app shows which artifact is absent instead of crashing.

## Tabs

1. **Explorar mapa** — raw FIRMS detections over time (date-range + CCAA
   filters; cluster / heatmap toggle). No model.
2. **Plan A — tramos** — relative risk-band distribution for a chosen protocol.
3. **Plan B — tasa base** — empirical `P(≥1 detection | CCAA, day)` choropleth.
4. **Plan C — excedencia** — reliability curves per margin (5/10/20 km). The
   live point scorer (coordinates + Open-Meteo) is phase 2.
5. **Estado** — background-pool fill progress + provisional caveats.

## Provisionality

The background pool is still being filled, so every model-derived number is
**provisional** (see `docs/provisional-status.md`). Check progress with
`python scripts/background_pool_status.py`.
