# Wildfire Analysis & ML Pipeline / Análisis de Incendios Forestales y Pipeline de ML

---

# [ENG]

## Overview

End-to-end pipeline for ingesting, processing, enriching, and analyzing wildfire data. Combines **NASA FIRMS** fire hotspot detections (VIIRS + MODIS) with **CLCPlus Backbone** land cover and **Open-Meteo** historical weather to produce an analysis-ready dataset for ML modeling.

**RAW data** → Sensor merging → CLC enrichment → Weather enrichment → Processed dataset → ML

## Requirements

- **Python** >= 3.10
- **Operating System**: Windows / Linux / macOS

### Runtime Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation and analysis |
| `matplotlib` | Data visualization |
| `scikit-learn` | Machine learning |
| `rasterio` | GeoTIFF raster data reading |
| `geopandas` | Geospatial vector data |
| `shapely` | Geometric operations |
| `pyyaml` | YAML configuration parsing |
| `requests` | HTTP requests (Open-Meteo API) |

## Project Structure

```
wildfires-project/
├── configs/
│   ├── project.yaml                # Central configuration file
│   └── confidence_thresholds.yaml  # MODIS↔VIIRS confidence mapping rules
├── data/
│   ├── raw/
│   │   ├── firms/Spain/{2023,2024}/  # Raw FIRMS CSVs (one folder per sensor)
│   │   │   ├── MODIS/                 #   modis_2023.csv
│   │   │   └── VIIRS/                 #   VIIRS parent folder
│   │   │       ├── VIIRS S-NPP/       #     viirs_snpp_2023.csv
│   │   │       └── VIIRS NOAA-20/     #     viirs_noaa20_2023.csv
│   │   └── clcplus/Spain/2023-2025/  # CLCPlus GeoTIFF tiles
│   └── processed/
│       ├── merged/                  # VIIRS+MODIS merged output
│       ├── enriched/                # CLC + weather enriched datasets
│       └── predictions/             # Model prediction output
├── notebooks/
│   ├── 00_master.ipynb              # Main notebook (end-to-end walkthrough)
│   ├── 01_data_exploration.ipynb    # Raw data exploration
│   ├── 02_enrichment.ipynb          # Enrichment process walkthrough
│   ├── 03_analysis.ipynb            # Statistical analysis + feature engineering
│   └── 04_modeling.ipynb            # ML modeling
├── src/wildfire/                    # Main Python package
│   ├── config.py                    # Configuration loading
│   ├── data/                        # Data loading (FIRMS, CLCPlus, Open-Meteo)
│   ├── geo/                         # Geospatial utilities (tile selection)
│   ├── enrichment/                  # Enrichment (merge, CLC, weather)
│   └── processing/                  # Processing (confidence mapping, validation)
├── scripts/                         # CLI scripts for enrichment pipelines
├── tests/                           # Test suite
└── versioning/
    └── catalog.yaml                 # Dataset versioning catalog
```

## Setup & Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd wildfires-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install the project

```bash
pip install -e ".[dev]"
```

## Configuration

All project settings are defined in `configs/project.yaml`:

- **Data paths** (`data/raw`, `data/processed`)
- **FIRMS settings** (countries, years, sensors)
- **CLCPlus settings** (validity period, resolution)
- **Open-Meteo settings** (hourly variables, API URL)
- **Output paths** (merged, enriched, predictions)

Confidence mapping rules are in `configs/confidence_thresholds.yaml`.

## How to Run

### Run the tests

```bash
pytest tests/
```

### Use the library

```python
from wildfire.data.firms import load_firms
from wildfire.enrichment.merge_sensors import merge_viirs_modis
from wildfire.enrichment.clc_enrichment import enrich_with_clc

# Load a single sensor
viirs = load_firms(country="Spain", year=2023, sensor="viirs_snpp")

# Merge VIIRS + MODIS
merged = merge_viirs_modis(country="Spain", years=[2023, 2024])

# Enrich with land cover
enriched = enrich_with_clc(merged)
```

### Run enrichment scripts

```bash
python scripts/merge_firms.py                    # Merge VIIRS + MODIS
python scripts/enrich_with_clc.py                # Add CLCPlus land cover
python scripts/enrich_with_weather.py --input enriched.csv  # Add weather
python scripts/build_dataset.py                  # Assemble final dataset
```

## Data Sources

| Dataset | Source | Format | Description |
|---|---|---|---|
| FIRMS | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | CSV | VIIRS (375m) + MODIS (1km) active fire detections |
| CLCPlus Backbone | [Copernicus Land Cover](https://land.copernicus.eu/) | GeoTIFF | Land cover classification at 10m resolution |
| Open-Meteo | [Open-Meteo](https://open-meteo.com/) | JSON API | Historical weather (temp, humidity, wind, precip, radiation) |

## Dataset Schema

The final dataset follows FIRMS columns expanded with enrichment:

- **FIRMS core**: `latitude`, `longitude`, `acq_date`, `acq_time`, `satellite`, `sensor`, `brightness`, `brightness_ir`, `scan`, `track`, `frp`, `daynight`
- **Confidence**: `confidence_cat`, `confidence_num`, `confidence_og_cat`, `confidence_og_num`
- **CLCPlus**: `clc_class`, `clc_label`
- **Weather**: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `shortwave_radiation`

## Architecture Principles

1. **Immutable raw data** — Raw datasets are never modified; all transformations produce new files.
2. **Layered processing** — Data loading (`data/`) → Enrichment (`enrichment/`) → Processing (`processing/`).
3. **Manual enrichment** — Enrichment scripts are run manually; no automated scheduling.
4. **Configuration-driven** — Paths, parameters, and source settings live in YAML configs.
5. **Modular notebooks** — One master notebook + supporting notebooks for specific tasks.

---

# [ESP]

## Descripción general

Pipeline completo para ingerir, procesar, enriquecir y analizar datos de incendios forestales. Combina detecciones de focos de calor de **NASA FIRMS** (VIIRS + MODIS) con **CLCPlus Backbone** (cobertura del suelo) y **Open-Meteo** (clima histórico) para producir un conjunto de datos listo para ML.

**Datos CRUDOS** → Fusión de sensores → Enriquecimiento CLC → Enriquecimiento climático → Dataset procesado → ML

## Requisitos

- **Python** >= 3.10
- **Sistema operativo**: Windows / Linux / macOS

### Dependencias en tiempo de ejecución

| Paquete | Propósito |
|---|---|
| `numpy` | Computación numérica |
| `pandas` | Manipulación y análisis de datos |
| `matplotlib` | Visualización de datos |
| `scikit-learn` | Aprendizaje automático |
| `rasterio` | Lectura de datos raster GeoTIFF |
| `geopandas` | Datos vectoriales geoespaciales |
| `shapely` | Operaciones geométricas |
| `pyyaml` | Análisis de configuración YAML |
| `requests` | Solicitudes HTTP (API Open-Meteo) |

## Estructura del proyecto

```
wildfires-project/
├── configs/
│   ├── project.yaml                # Archivo de configuración central
│   └── confidence_thresholds.yaml  # Reglas de mapeo de confianza MODIS↔VIIRS
├── data/
│   ├── raw/
│   │   ├── firms/Spain/{2023,2024}/  # CSVs FIRMS crudos (una carpeta por sensor)
│   │   │   ├── MODIS/                 #   modis_2023.csv
│   │   │   └── VIIRS/                 #   Carpeta padre VIIRS
│   │   │       ├── VIIRS S-NPP/       #     viirs_snpp_2023.csv
│   │   │       └── VIIRS NOAA-20/     #     viirs_noaa20_2023.csv
│   │   └── clcplus/Spain/2023-2025/  # Tiles GeoTIFF CLCPlus
│   └── processed/
│       ├── merged/                  # Salida combinada VIIRS+MODIS
│       ├── enriched/                # Datasets enriquecidos (CLC + clima)
│       └── predictions/             # Predicciones del modelo
├── notebooks/
│   ├── 00_master.ipynb              # Notebook principal
│   ├── 01_data_exploration.ipynb    # Exploración de datos crudos
│   ├── 02_enrichment.ipynb          # Proceso de enriquecimiento
│   ├── 03_analysis.ipynb            # Análisis estadístico
│   └── 04_modeling.ipynb            # Modelado ML
├── src/wildfire/                    # Paquete principal de Python
│   ├── config.py                    # Carga de configuración
│   ├── data/                        # Carga de datos (FIRMS, CLCPlus, Open-Meteo)
│   ├── geo/                         # Utilidades geoespaciales (selección de tiles)
│   ├── enrichment/                  # Enriquecimiento (fusión, CLC, clima)
│   └── processing/                  # Procesamiento (mapeo de confianza, validación)
├── scripts/                         # Scripts CLI para pipelines de enriquecimiento
├── tests/                           # Suite de pruebas
└── versioning/
    └── catalog.yaml                 # Catálogo de versionado de datasets
```

## Configuración e instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd wildfires-project
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Instalar el proyecto

```bash
pip install -e ".[dev]"
```

## Configuración

Toda la configuración del proyecto se define en `configs/project.yaml`:

- **Rutas de datos** (`data/raw`, `data/processed`)
- **Configuración FIRMS** (países, años, sensores)
- **Configuración CLCPlus** (período de validez, resolución)
- **Configuración Open-Meteo** (variables horarias, URL de la API)
- **Rutas de salida** (merged, enriched, predictions)

Las reglas de mapeo de confianza están en `configs/confidence_thresholds.yaml`.

## Cómo ejecutar

### Ejecutar las pruebas

```bash
pytest tests/
```

### Usar la librería

```python
from wildfire.data.firms import load_firms
from wildfire.enrichment.merge_sensors import merge_viirs_modis
from wildfire.enrichment.clc_enrichment import enrich_with_clc

# Cargar un solo sensor
viirs = load_firms(country="Spain", year=2023, sensor="viirs_snpp")

# Combinar VIIRS + MODIS
merged = merge_viirs_modis(country="Spain", years=[2023, 2024])

# Enriquecer con cobertura del suelo
enriched = enrich_with_clc(merged)
```

### Ejecutar scripts de enriquecimiento

```bash
python scripts/merge_firms.py                    # Combinar VIIRS + MODIS
python scripts/enrich_with_clc.py                # Añadir cobertura CLCPlus
python scripts/enrich_with_weather.py --input enriched.csv  # Añadir clima
python scripts/build_dataset.py                  # Ensamblar dataset final
```

## Fuentes de datos

| Dataset | Fuente | Formato | Descripción |
|---|---|---|---|
| FIRMS | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | CSV | Detecciones de fuego activo VIIRS (375m) + MODIS (1km) |
| CLCPlus Backbone | [Copernicus Land Cover](https://land.copernicus.eu/) | GeoTIFF | Clasificación de cobertura del suelo a 10m |
| Open-Meteo | [Open-Meteo](https://open-meteo.com/) | JSON API | Clima histórico (temp, humedad, viento, precip, radiación) |

## Esquema del dataset

El dataset final sigue las columnas de FIRMS expandidas con enriquecimiento:

- **Núcleo FIRMS**: `latitude`, `longitude`, `acq_date`, `acq_time`, `satellite`, `sensor`, `brightness`, `brightness_ir`, `scan`, `track`, `frp`, `daynight`
- **Confianza**: `confidence_cat`, `confidence_num`, `confidence_og_cat`, `confidence_og_num`
- **CLCPlus**: `clc_class`, `clc_label`
- **Clima**: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `shortwave_radiation`

## Principios de arquitectura

1. **Datos crudos inmutables** — Los conjuntos de datos crudos nunca se modifican; todas las transformaciones producen nuevos archivos.
2. **Procesamiento por capas** — Carga de datos (`data/`) → Enriquecimiento (`enrichment/`) → Procesamiento (`processing/`).
3. **Enriquecimiento manual** — Los scripts de enriquecimiento se ejecutan manualmente; no hay programación automatizada.
4. **Orientado a configuración** — Rutas, parámetros y ajustes de fuente están en archivos YAML.
5. **Notebooks modulares** — Un notebook principal + notebooks de soporte para tareas específicas.
