# Wildfire Dataset Project — Structure

```text
wildfire-dataset/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── firms/
│   │   │   └── .gitkeep
│   │   │
│   │   └── clcplus/
│   │       └── 2023/
│   │           ├── tile_01/
│   │           │   └── .gitkeep
│   │           ├── tile_02/
│   │           │   └── .gitkeep
│   │           └── ...
│   │
│   ├── intermediate/
│   │   ├── firms/
│   │   │   └── .gitkeep
│   │   └── clcplus/
│   │       └── .gitkeep
│   │
│   └── processed/
│       ├── fires/
│       │   └── .gitkeep
│       └── datasets/
│           └── .gitkeep
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_firms_exploration.ipynb
│   ├── 03_clc_exploration.ipynb
│   ├── 04_data_enrichment.ipynb
│   ├── 05_dataset_analysis.ipynb
│   └── 06_modeling.ipynb
│
├── src/
│   └── wildfire/
│       ├── __init__.py
│       ├── py.typed
│       │
│       ├── config.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── firms.py
│       │   ├── clc.py
│       │   └── loaders.py
│       │
│       ├── geo/
│       │   ├── __init__.py
│       │   ├── tiles.py
│       │   ├── raster.py
│       │   └── coordinates.py
│       │
│       ├── enrichment/
│       │   ├── __init__.py
│       │   └── clc_enrichment.py
│       │
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── cleaning.py
│       │   └── validation.py
│       │
│       └── analysis/
│           ├── __init__.py
│           └── statistics.py
│
├── scripts/
│   ├── prepare_firms.py
│   ├── enrich_with_clc.py
│   └── build_dataset.py
│
├── tests/
│   ├── test_firms.py
│   ├── test_clc.py
│   └── test_enrichment.py
│
├── configs/
│   └── project.yaml
│
└── versioning/
    └── .gitkeep
```

## Architectural principles

### 1. Raw data is immutable

`data/raw/` contains the original downloaded datasets and should not be modified.

Examples:

- NASA FIRMS data
- CLCplus Backbone tiles
- Future external datasets

If processing needs to be repeated, start again from the raw data.

### 2. Processing is reproducible

Derived data belongs in:

- `data/intermediate/`
- `data/processed/`

These can be regenerated from the raw data using the project's scripts and modules.

### 3. Dataset preparation is separate from modeling

The CLC enrichment process is considered part of **dataset construction**, not part of the ML model.

Conceptually:

```text
RAW DATA
   │
   ├── FIRMS
   ├── CLCplus
   └── Future sources
          │
          ▼
   PREPARATION / ENRICHMENT
          │
          ▼
   PROCESSED DATASET
          │
          ▼
   ANALYSIS / FEATURE ENGINEERING
          │
          ▼
        MODELING
```

This allows CLC to be replaced or additional sources to be added without coupling them to the model.

### 4. Notebooks are interfaces, not the main codebase

Notebooks should primarily contain:

- exploration
- visualisation
- experimentation
- interpretation
- calls to reusable functions

Reusable functionality should live under `src/wildfire/`.

For example:

```python
from wildfire.data.firms import load_firms
from wildfire.enrichment.clc_enrichment import enrich_with_clc

fires = load_firms(...)
fires = enrich_with_clc(fires, ...)
```

This makes the transition from a notebook-based project to a proper Python application much easier.

### 5. Geospatial functionality is separated from data sources

`src/wildfire/geo/` contains reusable geographic operations such as:

- finding the tile containing a coordinate
- converting coordinates
- querying raster pixels
- working with CRS
- spatial operations

These should not be tied exclusively to CLC because the same functionality may later be needed for:

- elevation
- weather grids
- NDVI
- population rasters
- other satellite products

### 6. Scripts provide reproducible pipelines

The `scripts/` directory contains executable workflows.

For example:

```bash
python scripts/enrich_with_clc.py
```

The scripts should use functions from `src/wildfire/` rather than containing large amounts of duplicated implementation code.

This allows the same functionality to be used from both notebooks and command-line scripts.

### 7. Configuration should not be hardcoded

Project paths and important dataset settings belong in `configs/project.yaml`.

For example:

```yaml
data:
  raw: data/raw
  intermediate: data/intermediate
  processed: data/processed

clc:
  version: 2023
  resolution: 10
```

This makes it easier to change datasets, versions, or environments without modifying source code.

### 8. Data versioning is tracked

The `versioning/` directory stores metadata about dataset versions (checksums, download dates, source URLs).

This allows:
- verifying data integrity over time
- reproducing exact dataset states
- detecting when raw data has changed

```text
versioning/
   └── catalog.yaml
```

## Expected evolution

### Initial stage

```text
Notebook
   │
   ▼
src/wildfire/
   │
   ▼
Processed dataset
```

### Later

```text
CLI / scripts
       │
       ▼
src/wildfire/
       │
       ▼
Data pipeline
       │
       ▼
Dataset
       │
       ▼
ML pipeline
```

The structure is deliberately **modular without being prematurely complex**. New data sources or processing stages should be added when they are actually needed rather than creating empty modules for every possible future source.
