# Agent Instructions — Wildfire Analysis Pipeline

## Commands

```bash
# Install (requires GDAL/system libs for rasterio/geopandas)
pip install -e ".[dev]"    # dev extras: pytest, mypy, ruff, ipykernel, nbformat, nbclient

# Tests (401 collected; 399 run by default, 2 integration deselected via addopts)
pytest tests/                            # default: -m "not integration" (see pyproject)
pytest -m integration                    # only the 2 real-network tests (spends API quota!)
pytest tests/test_clc.py                 # single file
pytest tests/test_clc.py::TestResolveNeighbor::test_cross_north_boundary  # single method

# Lint / format / typecheck (no custom config in pyproject.toml — all defaults)
ruff check src/ tests/
ruff format src/ tests/
mypy src/

# Notebook (headless validation/execution; nbformat + nbclient in dev extras)
python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/01_recopilacion_datos_limpieza.ipynb', as_version=4))"

# Demo (Streamlit + Folium; optional extra). Run from the repo root.
pip install -e ".[demo]"
python scripts/fit_occurrence_model.py   # once: writes the live Plan-C bundle
streamlit run demo/app.py
ruff check demo/                         # app files; also lint demo/ with the rest
```

No Makefile, no CI, no pre-commit hooks. Commands above are the entire dev workflow.

Baselines (2026-10-18, refreshed — the repo grew since the 2026-10-03 counts):
`ruff check src/ tests/ scripts/` = **34 errors** (`src/ tests/` = 32),
`ruff format --check src/ tests/ scripts/` = **33 files**, `mypy src/` = **39 errors**
(all pre-existing: `typology.py`/`unsupervised.py`/pandas stubs). Touched files are
CLEAN (changed files pass `ruff check`/`ruff format --check`; mypy adds no new errors).
Don't add new ones; wholesale cleanup is tracked in TODO/compact, not done ad hoc.

## Architecture

Layered pipeline. Run manually via CLI scripts in `scripts/` or the notebook.

**Pipeline order (since 09-30, differs from README)**: RAW DATA → Sensor Merge →
Confidence Mapping → **Weather (batched)** → **CLC** → **geopandas/CCAA budgets** → Analysis/ML.

CSV chain under `data/processed/` — every name denotes its contents, each feeds the next:

1. `merged/firms_spain_merged.csv` — FIRMS + confidence (22 cols)
2. `enriched/firms_spain_weather_partial.csv` — +15 weather (47,505×37, `temperature_2m` complete; 8,912 rows lack `boundary_layer_height`, null upstream in 2024 responses)
3. `enriched/firms_spain_weather.csv` — completed weather (NOT published, by team decision: the loader falls back to the partial and notebook 01 imputes BLH with the median)
4. `enriched/firms_spain_weather_clc.csv` — + `clc_class`, neighbors, `clc_uniform_surroundings` (47,505×47)
5. `enriched/firms_spain_full_enrichment.csv` — + `ccaa`, `prevention_current`, `prevention_prev{N}`, `extinction_prev{N}` (full-enrichment output, 30,931×51 after the decided `clc!=1` + `type==0` filters; N = `ccaa.budget_lookback_years`, extinction excludes the current year — no leakage; `firms_spain_final.csv` is reserved for the post-exploration/dedup modeling input)

**Supervised branch (05, feeds only notebook 04)**: `expanded/firms_spain_final_lags_index.csv` (positives) + `expanded/background_days.csv` (negatives, built by `notebooks/02d_background_data.ipynb`) → notebook 04 occurrence ranking. This branch does **not** feed the FIRMS enrichment chain above; the two meet only in `expanded/`.

- `src/wildfire/data/` — FIRMS CSV loading (`firms.py`), CLCPlus tile listing (`clc.py`),
  Open-Meteo single-row client (`openmeteo.py`), regional budget CSV (`ccaa.py`)
- `src/wildfire/enrichment/` — `merge_sensors.py` (concat + confidence),
  `clc_enrichment.py` (tile index, neighbor resolution, pixel reads),
  `weather_batch.py` (**production weather path**: 500-row POST batches, resume CSV,
  429 classification/wait, `load_weather_for_clc`/`finalize_weather_csv`,
  `load_background_weather` — final→partial fallback for the supervised background),
  `weather_enrichment.py` (**old path**, 1 HTTP GET per row — superseded, kept for back-compat)
- `src/wildfire/processing/` — confidence mapping (`confidence_cat` fixed
  `Categorical ["l","n","h"]`), validation (warnings only, never raises;
  fail-closed: missing expected columns warn; covers all 15 weather vars,
  CLC neighbors, `ccaa`/sums via `validate_enriched_ccaa`), occurrence
  contract (`occurrence.py`: `EVERYDAY_FEATURES` allowlist, `CANDIDATE_FEATURES`
  reintegration candidates + `CANDIDATE_POSTERIOR_FEATURES` alternative encoding,
  blocked splits, PR-AUC), typology (`typology.py`
  incl. the elected k12: `fit_k12_typology`/`K12Typology`), reintegration
  (`reintegration.py`: `fit_background_scorer` — everything fit on positives
  only, background scored by the fitted objects; `score(..., include_posteriors=)`
  returns the 12 k12 GMM posteriors for the alternative encoding),
  `feature_engineering.py` (**ported notebook-01b formulas** shared by batch and
  inference: `SUP_FORESTAL_HA`, `CCAA_HOLIDAY_CODES`, `add_calendar_features`,
  `add_anthropogenic_features`, `add_cyclic_features`, `add_budget_features`,
  `add_meteo_indices`, `add_engineered_features`)
- `src/wildfire/analysis/` — supervised modeling units called by notebook 04
  (the experiment loop lives in the notebook): `occurrence_model.py` (model
  candidates incl. class-balanced variants, per-fold scaler, fold-fit one-hot,
  continuous/categorical policy incl. posteriors), `ablation.py`
  (`ablation_verdict` promotion gate), `deployment.py` (Plan-D deployment unit:
  `ccaa_day_fire_rates` empirical CCAA×day fire-rate table, `quantile_bands`/
  `add_risk_bands` relative 5-band triage, `join_ccaa_fire_rates` base-rate join),
  `calibration.py` (Plan-D route C, **provisional**: wide-margin exceedance
  probability — `margin_positive_labels` same-day `R`-km labeling,
  `fit_isotonic`/`apply_isotonic`, `prior_correct` odds rescale,
  `reliability_curve`/`calibration_metrics` (Brier/log-loss/ECE),
  `cross_fitted_isotonic` day-blocked cross-fit, `exceedance_base_rate`
  model-free background base rate + day-bootstrap CI, `dedupe_events`,
  `calibrate_exceedance`/`exceedance_summary` orchestrator; margins
  `EXCEEDANCE_MARGINS_KM` = {5,10,20} km), `model_bundle.py` (Plan-D route C
  deployment bundle: `fit_preprocessing`/`transform` mirror the notebook's
  `prepare_fold`, `OccurrenceBundle` = estimator + preprocessing + per-margin
  `MarginCalibrator` + metadata; `save_bundle`/`load_bundle` joblib,
  `predict_score`/`calibrated_exceedance`/`score_point`), `inference.py`
  (**live point scorer** for the demo: `build_point_features(lat, lon, date,
  hour)` chains `fetch_weather` → `interpolate_hourly_fields` → `enrich_with_clc`
  → `assign_ccaa` → `add_ccaa_budget_sums` → `add_engineered_features` →
  `occurrence_features`), `status.py` (read-only background-pool
  fill/ratio report — backs `scripts/background_pool_status.py` and the demo),
  `demo_data.py` (read-only demo loaders: raw detections, per-protocol scores,
  band counts, CCAA base-rate GeoJSON for the choropleth, reliability/fold tables)
- `demo/` — Streamlit + Folium viewer (`app.py`; optional `[demo]` extra). Reads
  the notebook 01/04 artifacts: raw-detection time map (cluster/heatmap), Plan A
  bands, Plan B CCAA base-rate choropleth, Plan C reliability curves **and a live
  point scorer** (coordinates + date + hour → `analysis/inference.py` → Open-Meteo
  → the persisted `output.model_bundle` → calibrated exceedance) and a
  status panel. The bundle is written by `scripts/fit_occurrence_model.py`
  and versioned at `data/processed/models/occurrence_bundle.joblib`. Five tabs
  between the fixed header (logo/title/description) and the content; each tab
  holds its own controls (Explore filters in the detections tab, protocol in
  the risk-bands tab), no sidebar. Theme: `PALETTE` in
  `analysis/demo_data.py` mirrored by `.streamlit/config.toml` (light corporate
  palette); the centered logo lives in `demo/assets/` (`st.image`).
- `src/wildfire/geo/` — CCAA boundaries (`regions.py`: `load_boundaries` with
  `make_valid`, `assign_ccaa` — `within` join in EPSG:3035 + `sjoin_nearest`
  fallback ≤ `ccaa.nearest_max_distance_m`, keep-first dedupe, `REGION_NAME_MAP`
  translating GeoJSON names to exact budget names)
- `configs/project.yaml` — all paths and parameters (no hardcoded paths in source)
- `configs/confidence_thresholds.yaml` — MODIS↔VIIRS confidence mapping rules

Raw data in `data/raw/` is never modified. All output goes to `data/processed/`.
All save/load helpers anchor to `PROJECT_ROOT` (Phase 4, 2026-10-02) so notebooks run
with cwd = `notebooks/` without splitting the dataset. `save_enriched(..., filename=)`
writes a named pipeline stage; `filename=None` keeps the legacy `firms_*_enriched.csv`.

## Testing

### Integration tests

`@pytest.mark.integration` is **registered** in `pyproject.toml` and `addopts = "-m 'not integration'"`
excludes it by default. CLI `-m` overrides addopts (last wins).

```bash
pytest -m integration         # only test_openmeteo + test_openmeteo_requests POST test
```

The POST integration test (`test_fetch_next_batch_fills_partial`) checks the
resumed CSV only — the HTTP response is **not saved to disk** anymore
(the `openmeteo_response.json` dump was removed 2026-10-17 as unused; the
request cache already prevents re-spending quota on retries). It still
spends real quota on a cache miss.

### Real data dependencies

Some tests load actual CSVs from disk — they will fail if data files are missing:

- `test_confidence.py::TestAgainstRealData` (7) — needs `data/processed/enriched/firms_spain_enriched.csv`
- `test_validation.py::TestAgainstRealData` (3) — same file
- `test_firms.py::TestLoadFirms/TestLoadAllFirms/TestListAvailableFirms` — needs `data/raw/firms/Spain/2023/`
- `test_ccaa.py` (5 of 16) — needs `data/raw/ccaa/Datos_CCAA_Presupuesto - Datos y Variables Model.csv`
- `test_regions.py` (8 of 23) — needs the **untracked** `data/raw/ccaa/spain-communities.geojson`
  (map/boundaries tests) and the **untracked** `data/processed/enriched/firms_spain_weather_clc.csv`
  (real-data smoke test); the budget-mapping test uses the tracked budget CSV above
- `test_openmeteo_requests.py` (7 unmarked) — 3 need `data/processed/merged/firms_spain_merged.csv`;
  the interpolation tests use a synthetic in-memory payload (no JSON on disk since 2026-10-17)

These are **not** marked as integration. If you see failures on a fresh checkout, these are why.

### Experiment tests

`@pytest.mark.experiment` is **registered** in `pyproject.toml` and excluded from
the default `addopts` (`not integration and not experiment`); run it explicitly
with `pytest -m experiment -s`. Currently only
`tests/test_temporal_ratio_experiment.py`, an opt-in modeling study on real data:
it sweeps background ratios {0.25, 0.5, 1.0, 2.0} under a strict **train = 2023,
validate = 2024** temporal split (negatives drawn per year) and prints a
PR-AUC/ROC-AUC table. It recombines the public units only — it does not touch the
pipeline or replace the blocked-CV baseline.

### Fixtures

- `tests/test_confidence.py` — `thresholds` (session-scoped, loads YAML), `enriched_df` (module-scoped, loads real CSV)
- `tests/test_validation.py` — `enriched_df` (module-scoped, loads real CSV)
- `tests/conftest.py` — only adds `src/` to `sys.path`; no shared fixtures

Helper `_make_firms_df()` is duplicated in `test_confidence.py`, `test_validation.py`,
and `test_clc.py` with different signatures (plus a differently-named variant in `test_ccaa.py`).

## Notebook

`notebooks/01_recopilacion_datos_limpieza.ipynb` is the pipeline notebook: one cell per step
(Celda 1 merge → Celda 2 weather [fully commented, run the script externally] →
Celda 3 `load_weather_for_clc` → Celda 4 CLC → Celda 5 decided filter [`clc_class != 1` + `type == 0`]
→ Celda 6 `assign_ccaa` + `add_ccaa_budget_sums` → `firms_spain_full_enrichment.csv`).
Comments are bilingual ES/EN; cells only orchestrate — acquisition logic >5 lines lives in `src/`.

`notebooks/04_aprendizaje_supervisado.ipynb` is the supervised occurrence model
(19 cells, bilingual): load positives + `background_days.csv` → `validate_background_days`
→ `sample_background` → `assemble_modeling_frame` + `add_season` → blocked baseline
→ candidate-model table (`model_candidates`, incl. balanced variants) → reintegration
(`fit_background_scorer` scores both classes) → ablation + negative controls
(`ablation_verdict`) → **fold audit of `season/2024-Q1`** (banned-column check,
trivial calendar/CCAA control, per-class profile, permutation) → outputs
(`occurrence_scores.csv` with `risk_band`/`risk_percentile`/`ccaa_fire_rate`,
fold-metrics CSV, `ccaa_day_fire_rates.csv`, PR curves, per-CCAA choropleth,
permutation importance, package versions) → **wide-margin exceedance calibration**
(section 8, PROVISIONAL: train 2023 → calibrate 2024, isotonic + prior correction,
cross-fitted reliability, margins {5,10,20} km → `calibration_reliability.csv`) →
methods note. Runs headless with `nbclient` (cwd = `notebooks/`); the
prerequisite background artifact is built by `notebooks/02d_background_data.ipynb`.

## Gotchas

- **`src/wildfire/geo/` exists since 2026-10-02** (`regions.py`: CCAA boundaries from
  the team-provided `data/raw/ccaa/spain-communities.geojson`). README/TODO still
  document a `geo/tiles.py` that was never built — CLC tile logic lives in
  `enrichment/clc_enrichment.py` (see README drift below).

- **README drift**: README still describes the old CLC→weather order, the non-existent `geo/`
  module, English notebook names (actual: `00_introduccion.ipynb`, …), and mangled script-list
  comments. AGENTS.md + `compact/2026-09-30_notebook01-pipeline.md` are the accurate docs.

- **Notebook names are Spanish**: docs may reference English names (`00_master.ipynb`) but actual
  files are Spanish (`00_introduccion.ipynb`).

- **No tool config sections** except `[tool.pytest.ini_options]` (markers + addopts).
  `[tool.ruff]`, `[tool.mypy]` are absent — tools run on defaults.

- **Bilingual docs**: `README.md`, `TODO.md`, `wildfire_project_structure.md` use `[ENG]`/`[ESP]`
  section markers. Commit messages are mixed English/Spanish.

- **CLCPlus tiles**: EPSG:3035, tile grid `E{xx}N{yy}` (coords ÷ 100,000 m), 10,000×10,000 px
  at 10 m. Row 0 = north edge; the N key grows northward (E31N21 is north of E31N20) —
  `_resolve_neighbor` crosses boundaries accordingly (N/S inversion fixed 2026-10-02;
  6 tests in `TestResolveNeighbor` encode the corrected expectations).

- **Weather**: production path is `weather_batch.py` (one POST per 500 rows, resume CSV,
  rate-limit waits). `weather_enrichment.py` (1 request/row) is legacy. The old per-row
  path is still exported and wired to `scripts/enrich_with_weather.py`.

- **`confidence_cat` dtype**: after `add_unified_confidence()`, this column is
  `pandas.Categorical` with categories `["l", "n", "h"]`.

- **`tests/conftest.py`** manually manipulates `sys.path` instead of relying on editable install.

- **CLC tile keys**: canonical zero-padded `E{XX}N{YY}` via `_tile_key`
  (`clc_enrichment.py`) — lookups built `E17N9` missed index key `E17N09`
  (102 eastern-Canaries rows lost all CLC; fixed 2026-10-02 + 8 regression
  tests in `test_clc.py`: `TestTileKey`, N09 neighbor crossings, N09 index
  key, mocked N09 enrich).

- **Weather run state**: `firms_spain_weather_partial.csv` is complete on
  `temperature_2m` (47,505/47,505; resume == len, nothing left to download).
  8,912 rows lack `boundary_layer_height` (null upstream in the 2024
  responses), so `finalize_weather_csv` (default `require_complete=True`)
  refuses to publish `firms_spain_weather.csv` — absent by team decision, not
  by delay. The loader falls back to the partial and notebook 01 imputes BLH
  with the median (Celda 2.4), so nothing downstream is blocked. Publishing
  with the gap, or backfilling BLH via the batch script (spends real API
  quota), both remain possible but undecided.

- **`background_days.csv` is a RAW artifact** (same units/convention as
  `firms_spain_final_lags_index.csv`), keyed by `background_id` (integer,
  unique). Built by `notebooks/02d_background_data.ipynb`, which also writes a
  descriptive z-scored copy `background_days_normalized.csv` (EDA-only, never
  an input — the occurrence model fits its scaler per fold). 02d drops the
  all-null satellite columns and the same-day leakage rows
  (`drop_leaking_background` with `leakage_max_km`), so the saved file passes
  `validate_background_days` as-is. The pool is **complete and frozen**
  (2026-10-25): 97,630 rows (2023: 48,622; 2024: 49,008), every year above
  `background_ratio: 2` (3.31× / 5.80×), so `sample_background` draws the full
  ratio (46,284). Check with `python scripts/background_pool_status.py`
  (read-only); runbook + caveats in `docs/background-pool-status.md`.

- **Occurrence model (block 04, phase 2)**: PCA provenance verified 2026-10-18 —
  `fit_weather_pca` reproduces `features_a.csv` PC1–7 (max |diff| ~1e-15, k=7,
  cumulative variance 0.9152; `tests/test_pca_weather.py::TestAgainstRealFeaturesA`),
  so `BackgroundScorer` refitting on positives stays in the persisted typology's
  space. `model_candidates` now returns 5 estimators (three families + balanced
  variants); `BackgroundScorer.score(..., include_posteriors=True)` exposes the 12
  k12 GMM posteriors (via `CANDIDATE_POSTERIOR_FEATURES`). Ablation verdicts are
  computed on the frozen pool (2026-10-25; `background_ratio: 2`); the candidate
  set is not promoted (Δ +0.0073, CI95% includes 0);
  `analysis/deployment.py::ccaa_day_fire_rates` reports the empirical Plan-D base
  rate alongside the ranking. See `planning/2026-10-18_supervised-training-phase2.md`.
