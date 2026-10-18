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
```

No Makefile, no CI, no pre-commit hooks. Commands above are the entire dev workflow.

Baselines (2026-10-03): `ruff check src/ tests/ scripts/` = **28 errors** (pre-existing),
`ruff format --check` = **16 files** (pre-existing), `mypy src/` = **14 errors** (pre-existing).
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
  reintegration candidates, blocked splits, PR-AUC), typology (`typology.py`
  incl. the elected k12: `fit_k12_typology`/`K12Typology`), reintegration
  (`reintegration.py`: `fit_background_scorer` — everything fit on positives
  only, background scored by the fitted objects)
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
  `validate_background_days` as-is. The pool (~23.9k) is below
  `background_ratio: 2` × 23,142, so `sample_background` returns the whole pool
  (~1.03:1) until more background is generated.
