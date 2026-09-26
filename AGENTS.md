# Agent Instructions — Wildfire Analysis Pipeline

## Commands

```bash
# Install (requires GDAL/system libs for rasterio/geopandas)
pip install -e ".[dev]"

# Tests (195 total, 195 pass)
pytest tests/                                    # all
pytest tests/test_clc.py                         # single file
pytest tests/test_clc.py::TestResolveNeighbor    # single class
pytest tests/test_clc.py::TestResolveNeighbor::test_cross_north_boundary  # single method

# Lint / format / typecheck (no custom config in pyproject.toml — all defaults)
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

No Makefile, no CI, no pre-commit hooks. Commands above are the entire dev workflow.

## Architecture

Layered pipeline: RAW DATA → Sensor Merge → Confidence Mapping → CLC Enrichment → Weather Enrichment → Analysis/ML. Run manually via CLI scripts in `scripts/`.

- `src/wildfire/data/` — FIRMS CSV loading, CLCPlus tile listing, Open-Meteo HTTP client
- `src/wildfire/enrichment/` — merge_sensors, clc_enrichment, weather_enrichment
- `src/wildfire/processing/` — confidence mapping, validation
- `configs/project.yaml` — all paths and parameters (no hardcoded paths in source)
- `configs/confidence_thresholds.yaml` — MODIS↔VIIRS confidence mapping rules

Raw data in `data/raw/` is never modified. All output goes to `data/processed/`.

## Testing

### Integration tests

`@pytest.mark.integration` is used in `test_openmeteo.py` (real HTTP call) but **no pyproject.toml marker config exists**. Filtering requires explicit flags:

```bash
pytest -m "not integration"   # skip integration tests
pytest -m integration         # run only integration tests
```

### Real data dependencies

Some tests load actual CSVs from disk — they will fail if data files are missing:

- `test_confidence.py::TestAgainstRealData` — needs `data/processed/enriched/firms_spain_enriched.csv`
- `test_validation.py::TestAgainstRealData` — same file
- `test_firms.py::TestLoadFirms` — needs FIRMS CSVs at `data/raw/firms/Spain/2023/`

These are **not** marked as integration. If you see failures on a fresh checkout, these are why.

### Fixtures

- `tests/test_confidence.py` — `thresholds` (session-scoped, loads YAML), `enriched_df` (module-scoped, loads real CSV)
- `tests/test_validation.py` — `enriched_df` (module-scoped, loads real CSV)
- `tests/conftest.py` — only adds `src/` to `sys.path`; no shared fixtures

Helper `_make_firms_df()` is duplicated in `test_confidence.py`, `test_validation.py`, and `test_clc.py` with different signatures.

## Gotchas

- **Missing `src/wildfire/geo/`**: README and `wildfire_project_structure.md` document `geo/tiles.py` and `geo/coordinates.py`, but this directory does not exist. The CLCPlus tile logic lives in `enrichment/clc_enrichment.py` (`_build_tile_index`, `_resolve_neighbor`). No code imports from `wildfire.geo`.

- **Notebook names are Spanish**: docs reference English names (`00_master.ipynb`) but actual files are Spanish (`00_introduccion.ipynb`).

- **No tool config sections**: `pyproject.toml` has no `[tool.ruff]`, `[tool.mypy]`, or `[tool.pytest.ini_options]`. All tools run on defaults.

- **Bilingual docs**: `README.md`, `TODO.md`, `wildfire_project_structure.md` use `[ENG]`/`[ESP]` section markers. Commit messages are mixed English/Spanish.

- **CLCPlus tiles**: EPSG:3035 projection, tile grid `E{xx}N{yy}` (coordinates ÷ 100,000m), 10,000×10,000px at 10m resolution.

- **Weather enrichment**: One HTTP request per fire detection row (no batching). Slow for large datasets.

- **`confidence_cat` dtype**: After `add_unified_confidence()`, this column is `pandas.Categorical` with categories `["l", "n", "h"]`.

- **`tests/conftest.py`** manually manipulates `sys.path` instead of relying on editable install.
