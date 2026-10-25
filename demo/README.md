# Wildfire demo (Streamlit + Folium)

Read-only viewer for the supervised occurrence model's deployment units
(PLAN §Plan D), plus a **live** Plan-C point scorer. The browse tabs read the
artifacts notebooks 01/04 already wrote; only the live scorer spends Open-Meteo
quota (one request per point you submit).

## Install & run

From the repository root:

```powershell
pip install -e ".[demo]"
python scripts/fit_occurrence_model.py   # once: writes the Plan-C bundle
streamlit run demo/app.py
```

The app anchors all paths to `PROJECT_ROOT`, so it can be launched from any
working directory. The browse tabs need these artifacts (regenerate by running
the matching notebook):

| Artifact | Notebook |
|---|---|
| `data/processed/expanded/firms_spain_final_lags_index.csv` | 05 / pipeline |
| `data/processed/predictions/occurrence_scores.csv` | 04 |
| `data/processed/predictions/ccaa_day_fire_rates.csv` | 04 |
| `data/processed/predictions/calibration_reliability.csv` | 04 |
| `data/processed/predictions/occurrence_fold_metrics.csv` | 04 |
| `data/raw/ccaa/spain-communities.geojson` | team-provided |

The **live Plan-C scorer** additionally needs the persisted bundle
`data/processed/models/occurrence_bundle.joblib` (gitignored, reproducible),
written by `scripts/fit_occurrence_model.py` (see `docs/calibration-C.md`).

If one is missing the app shows which artifact is absent instead of crashing.

## Tabs

1. **Explorar mapa** — raw FIRMS detections over time (date-range + CCAA
   filters; cluster / heatmap toggle). No model.
2. **Plan A — tramos** — relative risk-band distribution for a chosen protocol.
3. **Plan B — tasa base** — empirical `P(≥1 detection | CCAA, day)` choropleth.
4. **Plan C — excedencia** — reliability curves per margin (5/10/20 km) **and**
   the live scorer: a clickable map to pick the point (a click fills the lat/lon
   inputs), plus date + hour → Open-Meteo → bundled model → calibrated
   `P(≥1 detection within R km that day)`.
5. **Estado** — background-pool status + usage caveats.

Guía completa en español de cada pestaña y control: `docs/guia-demo.md`.

## Background pool

The negatives pool is **complete and frozen** (2026-10-25): 97,630 usable rows,
every year above the configured `background_ratio: 2` (2023: 3.31×, 2024:
5.80×). Model-derived numbers are final for that snapshot. Check the live number
with `python scripts/background_pool_status.py`; details in
`docs/background-pool-status.md`.

Caveats that are not about pool size: the model is a **ranking** (not a point
probability), the route-C exceedance probability is a wide-margin approximation
(5/10/20 km), and PR-AUC is inflated by the ~1:2 framing.
