# TODO — Wildfire Project

---

## [ENG] Completed

- [x] Restructure data directories with country/year hierarchy
- [x] Create `configs/confidence_thresholds.yaml` for MODIS↔VIIRS mapping
- [x] Implement `data/firms.py` — FIRMS CSV loading
- [x] Implement `data/clc.py` — CLCPlus GeoTIFF loading
- [x] Implement `data/openmeteo.py` — Open-Meteo API client
- [x] Implement `processing/confidence.py` — confidence mapping logic
- [x] Implement `processing/validation.py` — data quality checks
- [x] Implement `enrichment/merge_sensors.py` — VIIRS+MODIS merge
- [x] Implement `geo/tiles.py` — programmatic tile selection
- [x] Implement `enrichment/clc_enrichment.py` — land cover enrichment
- [x] Implement `enrichment/weather_enrichment.py` — weather enrichment
- [x] Create CLI scripts in `scripts/`
- [x] Create placeholder notebooks (00–04)
- [x] Update `__init__.py` files with exports
- [x] Update `pyproject.toml` with `requests` dependency
- [x] Update `README.md`

## [ESP] Completado

- [x] Restructurar directorios de datos con jerarquía país/año
- [x] Crear `configs/confidence_thresholds.yaml` para mapeo MODIS↔VIIRS
- [x] Implementar `data/firms.py` — carga de CSVs FIRMS
- [x] Implementar `data/clc.py` — carga de GeoTIFF CLCPlus
- [x] Implementar `data/openmeteo.py` — cliente API Open-Meteo
- [x] Implementar `processing/confidence.py` — lógica de mapeo de confianza
- [x] Implementar `processing/validation.py` — verificación de calidad
- [x] Implementar `enrichment/merge_sensors.py` — fusión VIIRS+MODIS
- [x] Implementar `geo/tiles.py` — selección programática de tiles
- [x] Implementar `enrichment/clc_enrichment.py` — enriquecimiento de cobertura del suelo
- [x] Implementar `enrichment/weather_enrichment.py` — enriquecimiento climático
- [x] Crear scripts CLI en `scripts/`
- [x] Crear notebooks placeholder (00–04)
- [x] Actualizar archivos `__init__.py` con exports
- [x] Actualizar `pyproject.toml` con dependencia `requests`
- [x] Actualizar `README.md`

---

## [ENG] Pending

### 1. Implement tests [IN PROGRESS]

- [x] `tests/test_config.py` — config loading, directory structure, module imports
- [x] `tests/test_firms.py` — FIRMS loading, sensor mapping, folder structure
- [ ] `tests/test_confidence.py` — confidence mapping edge cases
- [ ] `tests/test_clc.py` — tile loading, pixel reading
- [ ] `tests/test_enrichment.py` — merge, CLC enrichment, weather enrichment

### 2. Download and place raw data [DONE]

- [x] FIRMS CSVs placed in `data/raw/firms/Spain/{2023,2024}/`
- [x] CLCPlus tiles placed in `data/raw/clcplus/Spain/2023-2025/`

### 3. Fix known bugs [DONE]

- [x] `load_firms()` — moved sensor validation before file existence check
- [x] `list_available_firms()` — use relative path from year dir for sensor key lookup
- [ ] `build_dataset.py` references `config["output"]["processed"]` which doesn't exist in `project.yaml`

### 4. Populate `versioning/catalog.yaml` [MEDIUM]

After first data download, add checksums and download dates.

### 4. Remove old directories if present [LOW]

Verify no stale `.gitkeep` or old directories remain in `data/`.

### 5. Define prediction target [MEDIUM]

TBD — the model task (binary classification, risk scoring, etc.) will be
determined after exploring the enriched dataset.

### 6. Explore extended weather indices [LOW]

Reminder to investigate VPD, drought index, and fire weather index in
`03_analysis.ipynb`.

### 7. Git branching strategy [MEDIUM]

With 4–6 contributors, establish:

- Feature branch naming convention (`feature/xxx`, `fix/xxx`)
- Code review process (PRs to `main`)
- Module ownership per person

### 8. Clean up obsolete files [DONE]

- [x] Remove stray executables from `scripts/` (venv artifacts)
- [x] Added `scripts/*.exe` to `.gitignore`

### 9. Evaluate CLI entry points [LOW]

When scripts mature, expose as CLI commands via `pyproject.toml`:

```toml
[project.scripts]
wildfire-merge = "wildfire.cli:merge_firms"
wildfire-enrich-clc = "wildfire.cli:enrich_with_clc"
wildfire-enrich-weather = "wildfire.cli:enrich_with_weather"
wildfire-build = "wildfire.cli:build_dataset"
```

---

## [ESP] Pendiente

### 1. Implementar tests [EN PROCESO]

- [x] `tests/test_config.py` — carga de configuración, estructura de directorios, imports
- [x] `tests/test_firms.py` — carga FIRMS, mapeo de sensores, estructura de carpetas
- [ ] `tests/test_confidence.py` — casos extremos de mapeo de confianza
- [ ] `tests/test_clc.py` — carga de tiles, lectura de píxeles
- [ ] `tests/test_enrichment.py` — fusión, enriquecimiento CLC, enriquecimiento climático

### 2. Descargar y colocar datos crudos [HECHO]

- [x] CSVs FIRMS colocados en `data/raw/firms/Spain/{2023,2024}/`
- [x] Tiles CLCPlus colocados en `data/raw/clcplus/Spain/2023-2025/`

### 3. Corregir bugs conocidos [HECHO]

- [x] `load_firms()` — validación del sensor movida antes de la verificación del archivo
- [x] `list_available_firms()` — usar ruta relativa desde directorio de año para buscar sensor
- [ ] `build_dataset.py` referencia `config["output"]["processed"]` que no existe en `project.yaml`

### 4. Poblar `versioning/catalog.yaml` [MEDIO]

Después de la primera descarga de datos, añadir checksums y fechas de descarga.

### 5. Eliminar directorios antiguos si existen [BAJO]

Verificar que no queden `.gitkeep` obsoletos o directorios antiguos en `data/`.

### 5. Definir el objetivo de predicción [MEDIO]

Por definir — la tarea del modelo (clasificación binaria, puntuación de riesgo, etc.)
se determinará después de explorar el dataset enriquecido.

### 6. Explorar índices climáticos extendidos [BAJO]

Recordatorio para investigar VPD, índice de sequía e índice de clima para incendios
en `03_analysis.ipynb`.

### 7. Estrategia de branching en Git [MEDIO]

Con 4–6 contribuyentes, establecer:

- Convención de nombres de branches (`feature/xxx`, `fix/xxx`)
- Proceso de revisión de código (PRs a `main`)
- Responsabilidad de módulo por persona

### 8. Limpiar archivos obsoletos [HECHO]

- [x] Eliminar ejecutables sueltos de `scripts/` (artefactos de venv)
- [x] Añadir `scripts/*.exe` a `.gitignore`

### 9. Evaluar entry points CLI [BAJO]

Cuando los scripts maduren, exponer como comandos CLI vía `pyproject.toml`:

```toml
[project.scripts]
wildfire-merge = "wildfire.cli:merge_firms"
wildfire-enrich-clc = "wildfire.cli:enrich_with_clc"
wildfire-enrich-weather = "wildfire.cli:enrich_with_weather"
wildfire-build = "wildfire.cli:build_dataset"
```
