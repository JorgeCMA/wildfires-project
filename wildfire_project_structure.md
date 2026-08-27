# Wildfire Dataset Project — Structure / Estructura del Proyecto de Dataset de Incendios Forestales

---

# [ENG]

## Project Tree

```text
wildfire-project/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── project.yaml                     # Central configuration
│   └── confidence_thresholds.yaml       # MODIS↔VIIRS confidence mapping
│
├── data/
│   ├── raw/
│   │   ├── firms/
│   │   │   └── Spain/
│   │   │       ├── 2023/
│   │   │       │   ├── MODIS/
│   │   │       │   │   └── modis_2023.csv
│   │   │       │   └── VIIRS/
│   │   │       │       ├── VIIRS S-NPP/
│   │   │       │       │   └── viirs_snpp_2023.csv
│   │   │       │       └── VIIRS NOAA-20/
│   │   │       │           └── viirs_noaa20_2023.csv
│   │   │       └── 2024/
│   │   │           ├── MODIS/
│   │   │           └── VIIRS/
│   │   │               ├── VIIRS S-NPP/
│   │   │               └── VIIRS NOAA-20/
│   │   │
│   │   └── clcplus/
│   │       └── Spain/
│   │           └── 2023-2025/
│   │               ├── Orignal_Data_99530.zip  # Original download (not tracked)
│   │               ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00/
│   │               │   ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.tif
│   │               │   ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.xml
│   │               │   └── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.tif.aux.xml
│   │               ├── ...              # tile directories
│   │               └── .gitkeep
│   │
│   └── processed/
│       ├── merged/
│       │   ├── firms_spain_2023_merged.csv
│       │   └── firms_spain_2024_merged.csv
│       ├── enriched/
│       │   ├── firms_spain_2023_enriched.csv
│       │   └── firms_spain_2024_enriched.csv
│       └── predictions/
│           └── .gitkeep
│
├── notebooks/
│   ├── 00_master.ipynb                  # Main: explains everything
│   ├── 01_data_exploration.ipynb        # FIRMS + CLCPlus + Open Meteo exploration
│   ├── 02_enrichment.ipynb              # Enrichment process walkthrough
│   ├── 03_analysis.ipynb                # Statistical analysis + feature engineering
│   └── 04_modeling.ipynb                # ML modeling
│
├── src/
│   └── wildfire/
│       ├── __init__.py
│       ├── py.typed
│       │
│       ├── config.py                    # Loads configs/project.yaml
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── firms.py                 # Load FIRMS CSVs by country/year/sensor
│       │   ├── clc.py                   # Load CLCPlus GeoTIFF tiles
│       │   └── openmeteo.py             # Open Meteo API client
│       │
│       ├── geo/
│       │   ├── __init__.py
│       │   ├── tiles.py                 # Programmatic CLCPlus tile selection
│       │   └── coordinates.py           # CRS conversion, coordinate utils
│       │
│       ├── enrichment/
│       │   ├── __init__.py
│       │   ├── merge_sensors.py         # Merge VIIRS + MODIS
│       │   ├── clc_enrichment.py        # Enrich with CLCPlus land cover
│       │   └── weather_enrichment.py    # Enrich with Open Meteo weather
│       │
│       └── processing/
│           ├── __init__.py
│           ├── confidence.py            # MODIS↔VIIRS confidence mapping
│           └── validation.py            # Data quality checks
│
├── scripts/
│   ├── merge_firms.py                   # Merge VIIRS + MODIS
│   ├── enrich_with_clc.py               # Run CLCPlus enrichment
│   ├── enrich_with_weather.py           # Run weather enrichment
│   └── build_dataset.py                 # Assemble final dataset
│
├── tests/
│   ├── test_firms.py
│   ├── test_clc.py
│   ├── test_enrichment.py
│   └── test_confidence.py
│
└── versioning/
    └── catalog.yaml
```

## Architectural Principles

### 1. Raw data is immutable

`data/raw/` contains the original downloaded datasets and should not be modified.

- NASA FIRMS CSVs (VIIRS + MODIS)
- CLCPlus Backbone GeoTIFF tiles

If processing needs to be repeated, start again from the raw data.

### 2. Layered processing

Processing follows a clear pipeline:

```text
RAW DATA
   │
   ├── FIRMS (VIIRS + MODIS)
   ├── CLCPlus tiles
   └── Open-Meteo API
          │
          ▼
   SENSOR MERGE (VIIRS + MODIS → unified FIRMS)
          │
          ▼
   CONFIDENCE MAPPING (reconcile numerical + categorical)
          │
          ▼
   CLC ENRICHMENT (add land cover class)
          │
          ▼
   WEATHER ENRICHMENT (add historical weather)
          │
          ▼
   PROCESSED DATASET
          │
          ▼
   ANALYSIS / ML
```

### 3. Manual enrichment

Enrichment scripts are run manually, not on a schedule. Data is downloaded
yearly from NASA FIRMS and Copernicus; enrichment code is called when needed.

### 4. Notebooks are interfaces, not the main codebase

- `00_master.ipynb` — end-to-end walkthrough (main deliverable)
- `01–04` — supporting notebooks for specific tasks

Reusable functionality lives in `src/wildfire/`.

### 5. Configuration should not be hardcoded

Project paths and settings belong in `configs/project.yaml`.
Confidence mapping rules belong in `configs/confidence_thresholds.yaml`.

### 6. Data versioning is tracked

`versioning/catalog.yaml` stores checksums and download dates.

## Dataset Schema

### FIRMS core columns

| Column | Type | Description |
|---|---|---|
| `latitude` | float | Fire pixel latitude |
| `longitude` | float | Fire pixel longitude |
| `acq_date` | str | Acquisition date (YYYY-MM-DD) |
| `acq_time` | str | Acquisition time (HHMM) |
| `satellite` | str | Satellite identifier |
| `sensor` | str | `modis`, `viirs_snpp`, `viirs_noaa20`, or `viirs` |
| `brightness` | float | Brightness temperature (K) |
| `brightness_ir` | float | IR channel brightness temperature (K) |
| `scan` | float | Along-scan pixel size |
| `track` | float | Along-track pixel size |
| `frp` | float | Fire Radiative Power (MW) |
| `daynight` | str | D=Day, N=Night |

### Confidence columns

| Column | Type | Description |
|---|---|---|
| `confidence_cat` | str | Unified categorical: low/nominal/high |
| `confidence_num` | float | Unified numerical: 0–100 |
| `confidence_og_num` | float | Original MODIS numerical 0–100 (NaN for VIIRS) |
| `confidence_og_cat` | str | Original VIIRS categorical l/n/h (NaN for MODIS) |

### CLCPlus enrichment columns

| Column | Type | Description |
|---|---|---|
| `clc_class` | int | CLC land cover class code |
| `clc_label` | str | Human-readable land cover label |

### Weather enrichment columns

| Column | Type | Description |
|---|---|---|
| `temperature_2m` | float | Temperature at 2m (°C) |
| `relative_humidity_2m` | float | Relative humidity at 2m (%) |
| `wind_speed_10m` | float | Wind speed at 10m (km/h) |
| `wind_direction_10m` | float | Wind direction at 10m (degrees) |
| `precipitation` | float | Precipitation (mm) |
| `shortwave_radiation` | float | Shortwave radiation (W/m²) |

---

# [ESP]

## Árbol del proyecto

```text
wildfire-project/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── project.yaml                     # Configuración central
│   └── confidence_thresholds.yaml       # Mapeo de confianza MODIS↔VIIRS
│
├── data/
│   ├── raw/
│   │   ├── firms/
│   │   │   └── Spain/
│   │   │       ├── 2023/
│   │   │       │   ├── MODIS/
│   │   │       │   │   └── modis_2023.csv
│   │   │       │   └── VIIRS/
│   │   │       │       ├── VIIRS S-NPP/
│   │   │       │       │   └── viirs_snpp_2023.csv
│   │   │       │       └── VIIRS NOAA-20/
│   │   │       │           └── viirs_noaa20_2023.csv
│   │   │       └── 2024/
│   │   │           ├── MODIS/
│   │   │           └── VIIRS/
│   │   │               ├── VIIRS S-NPP/
│   │   │               └── VIIRS NOAA-20/
│   │   │
│   │   └── clcplus/
│   │       └── Spain/
│   │           └── 2023-2025/
│   │               ├── Orignal_Data_99530.zip  # Descarga original (no tracked)
│   │               ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00/
│   │               │   ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.tif
│   │               │   ├── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.xml
│   │               │   └── CLMS_CLCPLUS_RAS_S2023_R10m_E15N10_03035_V01_R00.tif.aux.xml
│   │               ├── ...              # directorios de tiles
│   │               └── .gitkeep
│   │
│   └── processed/
│       ├── merged/
│       │   ├── firms_spain_2023_merged.csv
│       │   └── firms_spain_2024_merged.csv
│       ├── enriched/
│       │   ├── firms_spain_2023_enriched.csv
│       │   └── firms_spain_2024_enriched.csv
│       └── predictions/
│           └── .gitkeep
│
├── notebooks/
│   ├── 00_master.ipynb                  # Principal: explica todo
│   ├── 01_data_exploration.ipynb        # Exploración FIRMS + CLCPlus + Open Meteo
│   ├── 02_enrichment.ipynb              # Proceso de enriquecimiento
│   ├── 03_analysis.ipynb                # Análisis estadístico + ingeniería de características
│   └── 04_modeling.ipynb                # Modelado ML
│
├── src/
│   └── wildfire/
│       ├── __init__.py
│       ├── py.typed
│       │
│       ├── config.py                    # Carga configs/project.yaml
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── firms.py                 # Carga CSVs FIRMS por país/año/sensor
│       │   ├── clc.py                   # Carga tiles GeoTIFF CLCPlus
│       │   └── openmeteo.py             # Cliente API Open Meteo
│       │
│       ├── geo/
│       │   ├── __init__.py
│       │   ├── tiles.py                 # Selección programática de tiles CLCPlus
│       │   └── coordinates.py           # Conversión CRS, utilidades de coordenadas
│       │
│       ├── enrichment/
│       │   ├── __init__.py
│       │   ├── merge_sensors.py         # Combina VIIRS + MODIS
│       │   ├── clc_enrichment.py        # Enriquece con cobertura del suelo CLCPlus
│       │   └── weather_enrichment.py    # Enriquece con clima Open Meteo
│       │
│       └── processing/
│           ├── __init__.py
│           ├── confidence.py            # Mapeo de confianza MODIS↔VIIRS
│           └── validation.py            # Verificación de calidad de datos
│
├── scripts/
│   ├── merge_firms.py                   # Combina VIIRS + MODIS
│   ├── enrich_with_clc.py               # Ejecuta enriquecimiento CLCPlus
│   ├── enrich_with_weather.py           # Ejecuta enriquecimiento climático
│   └── build_dataset.py                 # Ensambla dataset final
│
├── tests/
│   ├── test_firms.py
│   ├── test_clc.py
│   ├── test_enrichment.py
│   └── test_confidence.py
│
└── versioning/
    └── catalog.yaml
```

## Principios de Arquitectura

### 1. Los datos crudos son inmutables

`data/raw/` contiene los conjuntos de datos originales descargados y no deben ser modificados.

- CSVs de NASA FIRMS (VIIRS + MODIS)
- Tiles GeoTIFF de CLCPlus Backbone

Si el procesamiento necesita repetirse, comenzar de nuevo desde los datos crudos.

### 2. Procesamiento por capas

El procesamiento sigue un pipeline claro:

```text
DATOS CRUDOS
   │
   ├── FIRMS (VIIRS + MODIS)
   ├── Tiles CLCPlus
   └── API Open-Meteo
          │
          ▼
   FUSIÓN DE SENSORES (VIIRS + MODIS → FIRMS unificado)
          │
          ▼
   MAPEO DE CONFIANZA (reconciliar numérico + categórico)
          │
          ▼
   ENRIQUECIMIENTO CLC (añadir clase de cobertura del suelo)
          │
          ▼
   ENRIQUECIMIENTO CLIMÁTICO (añadir clima histórico)
          │
          ▼
   DATASET PROCESADO
          │
          ▼
   ANÁLISIS / ML
```

### 3. Enriquecimiento manual

Los scripts de enriquecimiento se ejecutan manualmente, no en un horario.
Los datos se descargan anualmente de NASA FIRMS y Copernicus; el código de
enriquecimiento se llama cuando es necesario.

### 4. Los notebooks son interfaces, no el código principal

- `00_master.ipynb` — recorrido completo (entregable principal)
- `01–04` — notebooks de soporte para tareas específicas

La funcionalidad reutilizable vive en `src/wildfire/`.

### 5. La configuración no debe estar hardcodeada

Las rutas y configuraciones del proyecto pertenecen a `configs/project.yaml`.
Las reglas de mapeo de confianza pertenecen a `configs/confidence_thresholds.yaml`.

### 6. El versionado de datos se rastrea

`versioning/catalog.yaml` almacena checksums y fechas de descarga.

## Esquema del Dataset

### Columnas principales de FIRMS

| Columna | Tipo | Descripción |
|---|---|---|
| `latitude` | float | Latitud del píxel de fuego |
| `longitude` | float | Longitud del píxel de fuego |
| `acq_date` | str | Fecha de adquisición (AAAA-MM-DD) |
| `acq_time` | str | Hora de adquisición (HHMM) |
| `satellite` | str | Identificador del satélite |
| `sensor` | str | `viirs_snpp`, `modis`, `viirs_noaa20` o `viirs` |
| `brightness` | float | Temperatura de brillo (K) |
| `brightness_ir` | float | Temperatura de brillo canal IR (K) |
| `scan` | float | Tamaño del píxel en escaneo |
| `track` | float | Tamaño del píxel en seguimiento |
| `frp` | float | Potencia Radiativa del Fuego (MW) |
| `daynight` | str | D=Día, N=Noche |

### Columnas de confianza

| Columna | Tipo | Descripción |
|---|---|---|
| `confidence_cat` | str | Categórica unificada: low/nominal/high |
| `confidence_num` | float | Numérica unificada: 0–100 |
| `confidence_og_cat` | str | Categórica original VIIRS l/n/h (NaN para MODIS) |
| `confidence_og_num` | float | Numérica original MODIS 0–100 (NaN para VIIRS) |

### Columnas de enriquecimiento CLCPlus

| Columna | Tipo | Descripción |
|---|---|---|
| `clc_class` | int | Código de clase de cobertura del suelo CLC |
| `clc_label` | str | Etiqueta legible de cobertura del suelo |

### Columnas de enriquecimiento climático

| Columna | Tipo | Descripción |
|---|---|---|
| `temperature_2m` | float | Temperatura a 2m (°C) |
| `relative_humidity_2m` | float | Humedad relativa a 2m (%) |
| `wind_speed_10m` | float | Velocidad del viento a 10m (km/h) |
| `wind_direction_10m` | float | Dirección del viento a 10m (grados) |
| `precipitation` | float | Precipitación (mm) |
| `shortwave_radiation` | float | Radiación de onda corta (W/m²) |
