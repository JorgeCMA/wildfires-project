# Agent Instructions — Wildfire Analysis Pipeline

## Commands

```bash
# Install (requires GDAL/system libs for rasterio/geopandas)
pip install -e ".[dev]"    # dev extras: pytest, mypy, ruff, ipykernel, nbformat, nbclient

# Tests (307 total; 305 run by default, 2 integration deselected via addopts)
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

Baselines (2026-10-02): `ruff check src/ tests/ scripts/` = **38 errors** (pre-existing),
`ruff format --check` = **17 files** (pre-existing), `mypy src/` = **14 errors** (pre-existing).
Don't add new ones; wholesale cleanup is tracked in TODO/compact, not done ad hoc.

## Architecture

Layered pipeline. Run manually via CLI scripts in `scripts/` or the notebook.

**Pipeline order (since 09-30, differs from README)**: RAW DATA → Sensor Merge →
Confidence Mapping → **Weather (batched)** → **CLC** → **geopandas/CCAA budgets** → Analysis/ML.

CSV chain under `data/processed/` — every name denotes its contents, each feeds the next:

1. `merged/firms_spain_merged.csv` — FIRMS + confidence (22 cols)
2. `enriched/firms_spain_weather_partial.csv` — +15 weather, resume point (in-progress)
3. `enriched/firms_spain_weather.csv` — completed weather (written by `finalize_weather_csv`)
4. `enriched/firms_spain_weather_clc.csv` — + `clc_class`, neighbors, `clc_uniform_surroundings`
5. `enriched/firms_spain_final.csv` — + `ccaa`, `sum_prevention`, `sum_extinction` (modeling input; 46,500×50 as of 2026-10-02 — re-run Celdas 3→4→6 after the weather download completes for all 47,505 rows)

- `src/wildfire/data/` — FIRMS CSV loading (`firms.py`), CLCPlus tile listing (`clc.py`),
  Open-Meteo single-row client (`openmeteo.py`), regional budget CSV (`ccaa.py`)
- `src/wildfire/enrichment/` — `merge_sensors.py` (concat + confidence),
  `clc_enrichment.py` (tile index, neighbor resolution, pixel reads),
  `weather_batch.py` (**production weather path**: 500-row POST batches, resume CSV,
  429 classification/wait, `load_weather_for_clc`/`finalize_weather_csv`),
  `weather_enrichment.py` (**old path**, 1 HTTP GET per row — superseded, kept for back-compat)
- `src/wildfire/processing/` — confidence mapping, validation (warnings only, never raises)
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

The POST integration test writes its response to `tmp_path` (no longer overwrites the
tracked `data/processed/openmeteo_response.json`). It still spends real quota on a cache miss.

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
  4 interpolation tests read the **tracked** `data/processed/openmeteo_response.json`

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
Celda 3 `load_weather_for_clc` → Celda 4 CLC → Celda 5 sealed TODO [drop `clc_class == 1`?]
→ Celda 6 `assign_ccaa` + `add_ccaa_budget_sums` → `firms_spain_final.csv`).
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

- **Weather run state**: `firms_spain_weather_partial.csv` resume = 46,500/47,505 (was
  39,500/47,505, then 32,500/47,505, stopped on the daily limit 2026-10-01; download still Jorge's job).
  Resume = re-run `python scripts/enrich_weather_batch.py`
  (exit 0 publishes `firms_spain_weather.csv`). After it completes, re-run notebook
  Celdas 3→4→6 to refresh `firms_spain_weather_clc.csv` + `firms_spain_final.csv`
  to all 47,505 rows — no code changes (`load_weather_for_clc` picks the completed
  file automatically).
