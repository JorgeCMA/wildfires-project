# Compact — Open-Meteo Batch Weather Enrichment (2026-09-28)

Replaces `compact/2026-09-20_test-quality-fixes.md` (its open items are absorbed into *Test Suite Status* and *Next Steps* below).

## ⚠️ Overall Status — STILL IN TEST, NOT IN PRODUCTION

Read this first. Everything about the new weather-batch feature is provisional:

1. **Implementation lives ONLY in `tests/test_openmeteo_requests.py`.** The test file doubles as the implementation harness: functions and tests are interleaved in the same module. Nothing has been moved to `src/wildfire/`.
2. **No new production code.** The only `src/` change in all sessions was *deleting* the dead helper `load_latest_merged_split` from `enrichment/merge_sensors.py` (+ its `__init__.py` export). `weather_enrichment.py` is untouched and still does its old one-HTTP-call-per-row flow.
3. **Not moved to the final folders.** The real enriched output `data/processed/enriched/firms_spain_enriched.csv` (7.3 MB) has **no weather columns** — it is still merged+CLC. Weather data exists only in `firms_spain_enriched_partial.csv`, which was generated manually/by the fetch function, is **untracked in git**, and is not wired into any pipeline script. Both data artifacts (`partial CSV`, `openmeteo_response.json`) are test/dev by-products, not committed deliverables.
4. **Pipeline order is decided but NOT applied in production**: merge → confidence → **open-meteo** → CLC. `firms_spain_enriched.csv` was built merged→CLC (old order). The reorder rationale: Open-Meteo has limited calls/day, so weather must come before CLC — CLC can then be re-run freely without re-spending weather quota.
5. **Tests are not correctly/finaly implemented yet**:
   - implementation + tests in the same file (should be split when promoted to `src/`);
   - `@pytest.mark.integration` is still **unregistered** (`PytestUnknownMarkWarning` ×2; no `[tool.pytest.ini_options]` in `pyproject.toml`);
   - integration test requires real network and is subject to 429 rate limits (one incident this session);
   - the deep validation was a **one-off report**, not permanent tests (`skipif`-guarded battery = possible follow-up, user chose report-first);
   - pre-existing real-data tests still fail on fresh checkouts (known AGENTS.md gotcha).

Uncommitted working tree: `tests/test_openmeteo_requests.py` (modified) + 2 untracked data files. Commits so far: `e639ae0 tests openmeteo`, `d1bf8b6 test open meteo`.

## Context

Goal: batch-download hourly weather for all **47,505** FIRMS Spain 2023 merged rows (22 columns) from Open-Meteo, with **resume** across calls, and write 15 interpolated weather columns per detection into a partial CSV. Runs in `tests/test_openmeteo_requests.py`; user explicitly chose "everything stays in the test file".

## Key Design Decisions

1. **Pipeline order**: merge → confidence → open-meteo → CLC (see status §4).
2. **Partial CSV base = merged** (22 cols), *not* the CLC-enriched file; weather columns appended progressively (15 new → 37 total).
3. **Resume sentinel = first row missing `temperature_2m`** (`find_resume_row`): column absent → 0; some NaN → its position; none → `len(df)`. All 15 columns are written together, so one sentinel suffices.
4. **Full CSV rewrite after every batch of 500** (crash-safe ordering: raw JSON saved first, then fill, then CSV — a failed fill just re-fetches the same cached batch).
5. **POST, not GET** (see API findings); `timezone: GMT` (UTC, matches FIRMS `acq_date`/`acq_time`).
6. **Integration test uses `tmp_path`** for the CSV → deterministic, never touches the real partial CSV; real progress only advances when `fetch_next_batch()` is called with defaults.

## API Findings (hard-won)

- **GET dies at 500 rows**: prepared URL 38,485 chars → nginx `414 Request-URI Too Large`. Same query as **POST → 200** (500 locations, 0.5 s, 424 KB for 1 var). Must call `openmeteo.weather_api(url, params=params, method="POST")`.
- Per-location `start_date`/`end_date` as Python lists (repeated query keys) works, incl. 500-row scale (54 distinct dates in first 500 rows).
- **Rate limit**: `{'error': True, 'reason': 'Minutely API request limit exceeded. Please try again in one minute.'}` → raised as `OpenMeteoRequestsError`. Batch 2 attempt hit it → stopped per instruction. Pacing ≈1 batch/min needed for remaining runs.
- **requests-cache does not cache POST by default** → added `allowable_methods=("GET", "POST")` (keys on URL+body) so repeated suite runs / same batch don't re-spend quota (fixed back-to-back 429 flakiness).
- **Timezone**: `auto` reported bogus `utc_offset_seconds=7200` for Europe/Madrid in January and shifted the flatbuffer window an hour early. With `GMT`: `UtcOffsetSeconds() == 0` asserted; flatbuffer `Timezone()` returns `None` → code uses `(response.Timezone() or b"GMT").decode()`.
- Flatbuffer epoch is absolute UTC; hourly window = 00:00–23:00 of `acq_date` in UTC; **hourly index == hour** (`[0]`=00:00 … `[23]`=23:00) — verified vs docs + saved JSON.
- `precipitation == rain` in 11,998/12,000 hours (differs only snowy hours) — API by design.
- Twilights: 4.1 % of hours have `is_day==0` while `shortwave_radiation>0` (max 57 W/m²) and 77 h day-but-0 at sunset edge — API physics, not our bug.

## Config Changes (`configs/project.yaml`, committed)

- `openmeteo.batch_rows: 500` — lives **outside** `batch_query` on purpose (comment: `batch_query` is copied verbatim into HTTP params → `batch_rows` must never be sent; test asserts this).
- `openmeteo.batch_query.timezone`: `auto` → `GMT` (comment: FIRMS times are UTC).
- Unchanged: 15 `hourly` vars, `models: best_match`, `base_url`.

## src Changes (committed)

- `enrichment/merge_sensors.py`: `load_latest_merged_split` deleted (−32 lines).
- `enrichment/__init__.py`: import + `__all__` entry removed (note: `RUF022 __all__ not sorted` is pre-existing and still flagged).

## Implemented Functions — `tests/test_openmeteo_requests.py` (line numbers = current working tree)

| Line | Function | Behavior |
|---|---|---|
| 22 | `ROW_KEYS` | `("latitude", "longitude", "start_date", "end_date")` — per-row params |
| 25 | `OUTPUT_JSON` | `data/processed/openmeteo_response.json` |
| 34 | `PARTIAL_CSV` | `data/processed/enriched/firms_spain_enriched_partial.csv` |
| 41 | `weather_fields()` | the 15 `batch_query.hourly` names, in order |
| 46 | `build_row_requests(df)` | one dict per row: lat/lon + `acq_date` as start==end + `batch_query` |
| 70 | `load_partial_or_merged(path=PARTIAL_CSV)` | partial CSV if exists else `load_merged()`; parametrized path |
| 83 | `find_resume_row(df)` | first row without `temperature_2m` (see design §3) |
| 98 | `next_batch(df, batch_rows)` | `df.iloc[start : start+batch_rows]`; defaults from config |
| 120 | `build_batch_request(df)` | `(url, params)`; ROW_KEYS → lists (one per location) + `batch_query` |
| 140 | `split_acq_time(acq_time)` | `HHMM` → `(hour, minutes, pct=m/60)`; pct = weight of **next** hour (1345 → 0.75) |
| 161 | `interpolate_hourly_fields(hourly, acq_time)` | index==hour, base read from `hourly[0]["date"][11:13]`; hour 23 clamps; `None` endpoint → `None` |
| 209 | `fill_weather(df, start, records)` | adds 15 float cols on first batch; `zip(..., strict=True)` (ValueError on mismatch); `None`→`NaN` (pandas rejects `None` for float64: `LossySetitemError`); one positional `iloc[...] = matrix` write; outside `[start, start+n)` untouched |
| 254 | `fetch_next_batch(save_path=OUTPUT_JSON, csv_path=PARTIAL_CSV)` | load → resume → POST → build `records` (24×15 via `Variables(i)`, `to_json` NaN→`null`) → save JSON payload `{row_start, url, params, responses}` → `fill_weather` → full `to_csv(index=False)` → prints filled range + resume |

### Test inventory (same file)

- Offline config/resume: `test_batch_rows_config` (381), `test_find_resume_row_*` (392/399/406), `test_next_batch_*` (413/422)
- Offline fill: `_synthetic_record` (443), `test_fill_weather_adds_columns_and_values` (454), `..._leaves_tail_nan_and_advances_resume` (468), `..._handles_none_values` (482), `..._length_mismatch_raises` (496)
- Offline requests: `test_row_requests_from_first_rows` (505), `test_build_batch_request_from_first_rows` (519)
- Integration: `test_fetch_next_batch_saves_response(tmp_path)` (542) — tmp CSV, asserts `row_start==0`, params == fresh `build_batch_request(batch)`, GMT/offset 0, first+last record coords/date/24h×15 fields, CSV = merged+15 cols, rows <n filled / ≥n NaN, `find_resume_row==n`, row-0 value matches recompute
- Offline interpolation (use saved JSON): `_payload`/`_saved_responses`/`_saved_rows` (604/610/615 — rows recovered via `payload["row_start"]`), `test_hourly_index_matches_documentation` (622), `test_split_acq_time` (636), `test_interpolate_matches_saved_response` (645, None branch included), `test_interpolate_concrete_value` (674), `test_interpolate_edge_cases` (687)

## Data Artifacts (all UNTRACKED)

| File | Size | State |
|---|---|---|
| `data/processed/merged/firms_spain_merged.csv` | 5.3 MB | 47,505 × 22 (tracked) |
| `data/processed/enriched/firms_spain_enriched_partial.csv` | 6.2 MB | 47,505 × 37; **rows 0–499 filled** (7,500/7,500 values), 500+ all NaN (705,075); `find_resume_row == 500` |
| `data/processed/openmeteo_response.json` | 8.7 MB | `row_start: 0`, 500 responses, 24 h × 15 vars; **overwritten every batch** (only last batch retained) |
| `.cache/` | — | requests-cache, TTL 1 h, POST bodies included |
| `data/processed/enriched/firms_spain_enriched.csv` | 7.3 MB | tracked, **still CLC-only, no weather** |

## Batch Run Status

- **Batch 1 (rows 0–499): done** — served from cache (same body POSTed during tests), wrote the real partial CSV.
- **Batch 2 (rows 500–999): attempted → minutely rate limit → stopped** (state consistent: JSON `row_start 0` + CSV resume 500; window condition `row_start+len==resume` holds).
- **94 batches remain** (~47,005 rows); resume is automatic: next `fetch_next_batch()` reads the partial CSV and continues at 500. Pace ≈1/min or back off after 429.

## Validation Report (one-off — user chose report first, tests as follow-up)

All PASS on the 500 filled rows:

- **Structure**: 6.2 MB, (47,505 × 37), column order `22 merged + 15 weather` ✅
- **Provenance**: `assert_frame_equal(df[merged.columns], merged)` — original 22 columns **identical** (values + dtypes) ✅
- **Fill pattern**: 7,500/7,500 present in rows 0–499; 705,075 NaN in 500+; resume==500 ✅
- **Fidelity**: 7,500 values vs independent JSON recompute → **max abs diff 4.547e-13**; boundedness violations **0/7,500** ✅
- **Ranges**: 0 violations across 14 envelopes (temp [-25,50], humidity [0,100], pressure [800,1060], dir [0,360], soil [0,1], no negatives, `is_day`∈{0,1}, gust≥speed) ✅
- **Cross-source**: FIRMS `daynight` vs interpolated `is_day` → **0/500 mismatches** ✅
- **Twilight XOR**: 496/12,000 h = 4.1 % (tolerance 5 %) ✅
- **Temp by month**: Jan mean 11.3 (0.2…21.0), Feb 11.5 (−1.0…20.3), Mar 11.1 (3.5…18.9) — plausible ✅
- **Pressure outliers explained**: rows 144/145/436/438 at 821–849 hPa = mountain detections, API elevations 1,518–1,717 m (Gredos 40.25/-5.69; León 42.2/-6.7) — standard-atmosphere consistent ✅
- **Resume**: `next_batch()` → rows 500–999 ✅
- **Spot prints**: rows 0 (acq 0221 → 12.6535), 250 (1315 → 19.3625), 499 (0202 → 10.5222) — hand-computed == CSV ✅

## Test Suite Status

- **214 passed** (full run, ×3 consecutive green after POST-caching fix); includes 2 integration tests.
- Progression: 195 (2026-09-20 compact) → 210 (batch refactor) → 214 (+4 `fill_weather` tests).
- `ruff check` + `ruff format --check` clean on our file; `mypy src/` = 14 pre-existing errors (unchanged); `ruff src/` has pre-existing errors (incl. `RUF022 __all__` in `enrichment/__init__.py`).
- Warnings: `PytestUnknownMarkWarning` ×2 (markers still unregistered — carried over from previous compact).
- Still open from old compact: unregistered `integration` marker, real-data tests unmarked (e.g. `test_firms.py::TestLoadFirms`), duplicated `_make_firms_df` helper.

## Gotchas Learned This Round

- **PowerShell 5.1 strips double quotes** inside `python -c` here-strings → use single quotes / `%`-formatting in inline scripts.
- **`openmeteo_requests.Client` is the class, not the module** → `except openmeteo_requests.Client.OpenMeteoRequestsError` raises `AttributeError`; correct: `from openmeteo_requests.Client import OpenMeteoRequestsError`. (This is why the batch-2 rate limit escaped uncaught in the driver.)
- **pandas rejects `None` in float64 `iloc` setitem** (`LossySetitemError` → `TypeError: Invalid value '[None]'`) → convert to `float("nan")` when building the fill matrix.
- `serializable.to_json(orient="records")` converts `NaN → null` → hourly dicts contain `None`; interpolation must handle it.
- **JSON is overwritten per batch** → fidelity/interpolation tests must slice rows by `payload["row_start"]`, never `head(3)`; fidelity check only valid when `row_start + len(responses) == find_resume_row`.
- `_saved_rows()` reads `load_partial_or_merged()` — now returns the partial CSV when present (same row order as merged, so results identical).
- Integration test MUST write to `tmp_path`: running it against the real `PARTIAL_CSV` would advance/duplicate real progress and break determinism.
- `test_openmeteo.py::TestFetchWeatherIntegration` does an **uncached plain GET** → can 429 if the suite is hammered twice within a minute (pre-existing; our POST is now cached).
- `to_csv`/`read_csv` float round-trip is precise to ~1e-13 (full repr), safe for weather values.
- `git` warning: `LF will be replaced by CRLF` on our files (line-ending autocrlf; cosmetic).

## Next Steps

1. **Finish the data**: pace remaining 94 batches via `fetch_next_batch()` (stops cleanly on 429).
2. **Promote out of the test file** → production module (e.g. `src/wildfire/enrichment/…`), split implementation from tests, add a CLI script under `scripts/`.
3. **Apply the pipeline reorder in production**: build `firms_spain_enriched.csv` as weather-then-CLC (currently still merged-then-CLC); move partial/final outputs to their final folders and wire `PARTIAL_CSV` into config/`output`.
4. **Permanent validation tests**: `skipif`-guarded structural/provenance/fidelity battery (user: one-off report now, tests as possible follow-up).
5. Register pytest markers in `pyproject.toml` (fixes `PytestUnknownMarkWarning`).
6. Decide tracking for the 2 untracked data artifacts (commit vs gitignore).
7. Update `AGENTS.md` (test counts 195→214, new file role, POST/414/429 facts).

## Key File References

- `tests/test_openmeteo_requests.py` — entire feature (functions 22–362, tests 381–695)
- `configs/project.yaml:38-51` — `batch_rows` + `batch_query` (GMT)
- `src/wildfire/enrichment/merge_sensors.py` — `load_latest_merged_split` removed
- `src/wildfire/enrichment/weather_enrichment.py` — untouched production path (per-row HTTP)
- `data/processed/enriched/firms_spain_enriched_partial.csv` — partial weather output (untracked)
- `data/processed/openmeteo_response.json` — last batch's raw response (untracked)
