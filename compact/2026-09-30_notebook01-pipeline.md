# Compact — Notebook 01 pipeline: weather → CLC → CCAA budgets (2026-09-30)

Replaces `compact/2026-09-28_openmeteo-batch-weather.md` (its open items are absorbed into *Carried-Over Open Items* below).

> **DRAFT — written up front for this session.** Status per part is updated as the parts land; Part 8 finalizes this file.

## Status by Part

| Part | Scope | State |
|---|---|---|
| 1 | This compact (draft + absorb 09-28) | ✅ |
| 2 | `ccaa:` config + `data/ccaa.py` + tests | ✅ |
| 3 | Weather loader/finalizer + partial rename + script | ✅ |
| 4 | `save_enriched(filename=…)` + **scope B: `PROJECT_ROOT`-anchor all 7 cwd-relative path sites** | 🔄 in progress |
| 5 | Notebook cells 1–5 (weather commented, CLC, sealed TODO) | ⬜ |
| 6 | Geopandas `assign_ccaa` + notebook cell 6 + final CSV | ⬜ (paused on boundaries-source question) |
| 7 | Full verification (ruff/mypy/pytest + notebook run) | ⬜ |
| 8 | Finalize this compact + AGENTS.md update | ⬜ |

## Context / Goal

Rebuild notebook `01_recopilacion_datos_limpieza.ipynb` as a clean, one-cell-per-step pipeline where **each step's CSV feeds the next**, and all data-acquisition code >5 lines lives in `src/wildfire/` modules (notebook cells only orchestrate + analyze). Cell comments bilingual ES/EN.

Pipeline order (as decided in the 09-28 session, now being applied): **merge → weather → CLC → geopandas/CCAA**.

## Decisions (session 2026-09-30)

1. **Sum column names**: `sum_prevention` and `sum_extinction` (the initially-written `sum_extiction` was a typo; corrected).
2. **Window semantics**: relative to the row's year — `[Y − n + 1 .. Y]` anchored to the year of `acq_date` (n=3 → 2024 sums 2022+2023+2024; 2023 sums 2021+2022+2023).
3. **n default = 3** (`ccaa.years_window` in `configs/project.yaml`).
4. **Completed weather file**: new `firms_spain_weather.csv`; `scripts/enrich_weather_batch.py` finalizes it (copies the partial) when `run_weather_enrichment` returns `COMPLETE`.
5. **Naming chain** — every CSV name denotes its contents; each feeds the next:

   | # | File (`data/processed/…`) | Adds |
   |---|---|---|
   | 1 | `merged/firms_spain_merged.csv` | FIRMS + confidence (22 cols) — exists |
   | 2 | `enriched/firms_spain_weather_partial.csv` | +15 weather, partial (renamed from `firms_spain_enriched_partial.csv`) |
   | 3 | `enriched/firms_spain_weather.csv` | completed weather (47,505 rows) |
   | 4 | `enriched/firms_spain_weather_clc.csv` | + `clc_class`, neighbors, `clc_uniform_surroundings` |
   | 5 | `enriched/firms_spain_final.csv` | + `ccaa`, `sum_prevention`, `sum_extinction` — **modeling input** |

6. **Stale `firms_spain_enriched.csv` stays untouched** — `test_confidence.py::TestAgainstRealData` and `test_validation.py::TestAgainstRealData` read it (merged+CLC, no weather).
7. **Geopandas details deferred**: how/where the CCAA boundaries come from is decided at Part 6 (new `src/wildfire/geo/regions.py` + `ccaa.boundaries_path` key).

## Notebook Cell Layout (target)

1. **Cell 1** (existing): `merge_viirs_modis` + `save_merged` — unchanged.
2. **Cell 2** — *fully commented*: `run_weather_enrichment()` batch loop; header says run `python scripts/enrich_weather_batch.py` outside the analysis (~2 h, resume, 429/daily limits). Inert under "Run All".
3. **Cell 3** — `load_weather_for_clc()` → completed file if present, else partial filtered to rows with `temperature_2m` + analysis prints.
4. **Cell 4** — `enrich_with_clc(df)` → `save_enriched(…, filename="firms_spain_weather_clc.csv")` + `clc_class` value_counts.
5. **Cell 5** — *fully commented*: `# TODO (ES/EN) pendiente con el equipo — ¿eliminar filas con clc_class == 1 ("Sealed")?`
6. **Cell 6** — `assign_ccaa` → `add_ccaa_budget_sums` → save `firms_spain_final.csv` + analysis.

## Verification Sweep — post-Part 3 (2026-09-30)

- `pytest -m "not integration"` → **270 passed**; `mypy src/` → 14 errors (**exact baseline**, none in touched files); `ruff check src/ tests/ scripts/` → 38 errors, **all pre-existing/untouched**; `ruff format --check` → 17 unformatted, **none ours**.
- Functional: exports resolve; `load_weather_for_clc()` → 25,500×37 (0 NaN); `finalize_weather_csv()` → `None` + creates nothing (incomplete); script `--help` OK; YAML/notebook JSON valid; full chain smoke (weather→CLC→budgets, 20 rows) ran with `FutureWarning` as error; sums hand-checked.
- Full `pytest tests/` (integration **included**) → **2 failed**: Open-Meteo **daily quota exhausted by the parallel download** (`429` + `Daily API request limit exceeded`) — environmental, documented, not code.
- Git: user committed `2154f1a` (weather work) + `cff00a5` (Parts 1–2 + compact); working tree = Part 3 + partial rename (` D` old tracked / `??` new → `git add -A` when committing).

## Medium-Importance Issues (elaborated 2026-09-30; NOT in TODO — light issues went to TODO.md)

1. **Integration tests quota-flaky**: `test_openmeteo.py:149` (plain uncached `requests.get`, can't be cached as written) + `test_openmeteo_requests.py:198` (POST, cached but a miss costs a real 500-row/weight-700 call = ~7% of daily budget). Both fail whenever the download has spent the quota (today). Hidden side effect: the POST test's `save_path` defaults to `OUTPUT_JSON` → **overwrites the tracked `openmeteo_response.json`**. Fix direction: register marker + `addopts = "-m 'not integration'"` + skip-on-429 + `session=` param for `fetch_weather` + `save_path=tmp_path` in the test.
3. *(numbering as originally listed)* **Notebook tooling absent**: only `ipykernel`/`ipython`/`jupyter_client` installed — no `nbformat`/`nbconvert`/`nbclient`, so no `nbformat.validate()` and no headless execution (Part 7 acceptance check needs it). Root cause found alongside: **cwd-relative save/load helpers** (see Part 4 scope B) already produced a 5.3 MB duplicate at `notebooks/data/processed/merged/` (written 29-Sep 11:53).
2. **Large generated data in git**: `cff00a5` carried 317k JSON lines + 7.9k CSV lines of regenerated data; every weather batch rewrites both tracked artifacts → repo bloat. Needs a tracking policy.
4. **README drift**: documents the old CLC→weather order, non-existent `geo/` module, English notebook names (`00_master.ipynb` vs `00_introduccion.ipynb`), mangled script-list comments.

## Part 4 — Scope B (decided 2026-09-30)

- `save_enriched(df, country=…, year=…, filename: str | None = None)` — `None` keeps the legacy `firms_spain_enriched.csv` (back-compat for `scripts/enrich_with_clc.py` + real-data tests); explicit name → `data/processed/enriched/<filename>`.
- Anchor **all 7 cwd-relative sites** to `PROJECT_ROOT`: `merge_sensors.save_merged`/`load_merged`, `clc_enrichment.save_enriched`/`load_enriched`, `weather_enrichment.save_weather_enriched`, `firms.list_available_firms`, `weather_batch` `CachedSession(".cache")`. (`Path.__truediv__` lets an absolute config value win, so it degrades gracefully.) Then delete the orphaned `notebooks/data/` copy.
- Rationale: Jupyter/nbconvert run with cwd = notebook dir → relative paths split the dataset (notebook reads `notebooks/data/…`, tests/scripts read repo root). Tests already use `PROJECT_ROOT`, so their behavior is unchanged.

## Part 2 — Implemented (2026-09-30)

- `configs/project.yaml`: `ccaa.path`, `ccaa.years_window: 3` (comments explain the window and the NaN-not-0 rule).
- **New** `src/wildfire/data/ccaa.py`:
  - `load_ccaa_budget(path=None)` — reads the real CSV with `decimal=","` (comma-decimal Spanish format, UTF-8, no BOM; 102 rows = 17 CCAA × 2020–2025).
  - `add_ccaa_budget_sums(df, n=None, budget=None)` — adds `sum_prevention`/`sum_extinction`; builds one (row × offset-year) candidate table, `merge` on (region, year), `groupby.sum(min_count=1)` so an empty window is **NaN, never 0**; `budget` param exists for test injection; `KeyError` if `ccaa`/`acq_date` missing (geopandas step not run), `ValueError` if `n < 1`.
  - Public constants: `REGION_COL`, `YEAR_COL`, `PREVENTION_COL`, `EXTINCTION_COL`, `SUM_PREVENTION_COL`, `SUM_EXTINCTION_COL`.
- `src/wildfire/data/__init__.py`: exports + `__all__` sorted (fixes pre-existing RUF022 there).
- **New** `tests/test_ccaa.py` — 16 tests: real file shape/columns/dtypes/unique keys, decimal-comma parsing (tmp CSV), window math (2023→2021-2023, 2024→2022-2024), clipping at budget start (n=5 from 2023 → only 2020-2023), n=1, default-n via monkeypatched `load_config`, unknown region → NaN (not 0), missing columns, invalid n, input/index preservation, mixed regions/years.
- Verified: `pytest tests/test_ccaa.py` **16 passed**; `mypy src/` back to **14 errors** (baseline — pandas import uses `# type: ignore[import-untyped]` like `weather_batch.py`); `ruff check`/`format` clean on touched files (32 ruff errors elsewhere = pre-existing).
- Smoke-checked vs real data: Andalucía/2024 → prev 319.0 (84+110+125), ext 322.0 (91+113+118) ✓.

## Part 3 — Implemented (2026-09-30)

- `weather_batch.py`:
  - `PARTIAL_CSV` renamed → `firms_spain_weather_partial.csv` (file moved on disk; live resume kept working — was 5,000 at session start, **25,500** when Part 3 landed, i.e. the download has been running in parallel).
  - New `WEATHER_COMPLETE_CSV` → `firms_spain_weather.csv`.
  - New `load_weather_for_clc(complete_path, partial_path)` — contract of notebook cell 3: completed file if exists (verbatim), else partial filtered to rows with `temperature_2m`; no weather at all → `RuntimeError` telling the user to run the Open-Meteo step (paths are parameters so tests never touch the real CSVs).
  - New `finalize_weather_csv(partial_path, complete_path)` — copy (not move: the partial stays the resume point) only when `find_resume_row == len(df)`; returns the path, `None` if incomplete (creates nothing), no-op if the completed file already exists (idempotent), `FileNotFoundError` if the partial is missing.
- `scripts/enrich_weather_batch.py`: on `WeatherBatchStatus.COMPLETE` → calls `finalize_weather_csv()` and prints the published path; docstring updated. Exit codes unchanged.
- `enrichment/__init__.py`: `finalize_weather_csv`, `load_weather_for_clc` exported (`__all__` still sorted).
- Tests: +8 in `test_weather_batch.py` (`TestLoadWeatherForClc`, `TestFinalizeWeatherCsv`) using the new `_weather_csv` helper — completed-wins, partial-filtered, no-column/all-NaN errors, copy-when-complete, incomplete→None, idempotent sentinel, missing-partial error.
- Verified: **270 passed, 2 deselected**; `mypy src/` = 14 (baseline); ruff/format clean on touched files.

## Weather Batch Run State (absorbed from 09-28, updated 2026-09-30)

- **Resume = 25,500 / 47,505 rows** in `firms_spain_weather_partial.csv` (renamed from `firms_spain_enriched_partial.csv`; 5,000 at session start → download running in parallel). Log: `logs/weather_batch.log`.
- 09-28 run stopped cleanly on the **daily** limit (`daily_stop`); hourly-limit auto-wait worked (105 s). Remaining ~44 batches at `batch_interval_seconds: 450`; restart = re-run the script (which now also publishes `firms_spain_weather.csv` on completion).
- `data/processed/openmeteo_response.json` holds only the last batch (overwritten per batch).
- Old production path `weather_enrichment.py` (1 HTTP call/row) still exists, superseded by `weather_batch.py`.

## Test Suite Status

- **270 passed, 2 deselected** (`-m "not integration"`) after Part 3 (2026-09-30). Baseline 214 (09-28) + 32 weather-batch promotion + 16 `test_ccaa` (Part 2) + 8 weather loader/finalizer (Part 3).
- Still open: `integration` marker unregistered (`PytestUnknownMarkWarning`); real-data tests unmarked (`test_firms.py::TestLoadFirms`, `test_confidence.py::TestAgainstRealData`, `test_validation.py::TestAgainstRealData`); `_make_firms_df` duplicated across 3 files (now + helpers in `test_ccaa.py`, named differently).

## Carried-Over Open Items (from 09-28 and earlier)

1. Finish weather data: remaining ~94 batches via `scripts/enrich_weather_batch.py` (Part 3 makes it write the completed file).
2. Register pytest markers in `pyproject.toml`.
3. Decide tracking/gitignore for untracked data artifacts (partial CSV, `openmeteo_response.json`).
4. Permanent `skipif`-guarded validation battery (09-28 report was one-off).
5. Update `AGENTS.md` (test counts, new modules, partial rename, pipeline order) — Part 8.
6. Old en route leftovers: `versioning/catalog.yaml` checksums, prediction target definition, git branching strategy (TODO.md).
7. Uncommitted tree at session start (before this session's edits): modified `project.yaml`, `01_*.ipynb`, `enrichment/__init__.py`, `tests/test_openmeteo_requests.py`; untracked `weather_batch.py`, `test_weather_batch.py`, `scripts/enrich_weather_batch.py`, `scripts/probe_openmeteo_weight.py`, `logs/`, `notebooks/data/`, this compact.

## Key File References

- `notebooks/01_recopilacion_datos_limpieza.ipynb` — pipeline notebook (cells per part above)
- `src/wildfire/data/ccaa.py` — budgets loader + window sums (Part 2, done)
- `configs/project.yaml:91-100` — `ccaa:` section
- `src/wildfire/enrichment/weather_batch.py` — batch download, resume, rate limits (Part 3 target)
- `scripts/enrich_weather_batch.py` — unattended driver (Part 3 adds finalization)
- `src/wildfire/enrichment/clc_enrichment.py` — `save_enriched` (Part 4 adds `filename=`)
- `data/raw/ccaa/Datos_CCAA_Presupuesto - Datos y Variables Model.csv` — 102 rows, comma decimals, UTF-8
