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

### 0. Enrichment pipeline [CURRENT DEVELOPMENT]

- [ ] **CLCPlus Backbone enrichment** — enrich FIRMS data with land cover classes using CLCPlus tiles
- [ ] **Open METEO enrichment** — enrich with weather data (after CLCPlus is stable)

### 0.1 CLCPlus enrichment — known issues [IN PROGRESS]

- [x] **NaN nodata crash** (`clc_enrichment.py`) — if `src.nodata` is `NaN`, the `val == src.nodata` check always fails (IEEE 754), and `int(val)` on NaN raises `ValueError`. Added `np.isnan(val)` guard in `_pixel_value`.
- [ ] **`None == None` in `clc_uniform_surroundings`** (`clc_enrichment.py`) — missing-data pixels compare as equal, marking them "uniform". Add `df["clc_class"].notna()` check to the boolean expression.
- [ ] **Off-by-one wrapping in `_resolve_neighbor` for ring 3** (`clc_enrichment.py`) — hardcodes `9999`/`0` which is wrong for ±2 offsets. Use modular arithmetic: `new_row % 10000`. Latent bug (ring 3 not enabled).
- [ ] **Hardcoded column names in `clc_uniform_surroundings`** (`clc_enrichment.py`) — dynamic suffixes are computed but the comparison hardcodes `N`, `S`, `W`, `E`. Should use the dynamic list.
- [ ] **Hardcoded tile dimension `10000`** (`clc_enrichment.py`) vs dynamic `reader.height/width` elsewhere. Extract to a named constant or pass as parameter.
- [x] **Dead import** (`clc_enrichment.py`) — `import numpy as np` was unused. Now used for `np.isnan` guard.
- [ ] **Dead function** (`clc_enrichment.py`) — `load_clc_classes()` is never called anywhere. Either use it or prefix with `_`.
- [x] **Wrong return type annotation** (`clc_enrichment.py`) — `_pixel_value` now correctly returns `-> int | None` (was returning `float` for non-integer values).
- [x] **Docstring inconsistency** (`clc_enrichment.py`) — updated to reflect that only `clc_class` columns are added (removed `clc_name` columns from output).
- [ ] **No `try/finally` for reader cleanup** (`clc_enrichment.py`) — file handles leak on exception. Wrap main loop in `try/finally`.
- [ ] **Missing exports** (`enrichment/__init__.py`) — `NEIGHBORHOOD_RINGS`, `CLC_LABELS`, `load_clc_classes` are public but not in `__all__`. Either export or prefix with `_`.
- [x] **Remove `clc_name` columns** — center pixel and neighbor name columns removed from `enrich_with_clc` output. Only `clc_class` (int) columns remain.
- [ ] **No tests for CLC enrichment** — zero coverage for `_resolve_neighbor`, `NEIGHBORHOOD_RINGS`, `enrich_with_clc` output schema, `clc_uniform_surroundings`.

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

### 0. Pipeline de enriquecimiento [DESARROLLO ACTUAL]

- [ ] **Enriquecimiento CLCPlus Backbone** — enriquecer datos FIRMS con clases de cobertura del suelo usando tiles CLCPlus
- [ ] **Enriquecimiento Open METeo** — enriquecer con datos climáticos (después de que CLCPlus sea estable)

### 0.1 Enriquecimiento CLCPlus — problemas conocidos [EN PROCESO]

- [x] **Crash por nodata NaN** (`clc_enrichment.py`) — si `src.nodata` es `NaN`, la comprobación `val == src.nodata` siempre falla (IEEE 754) y `int(val)` sobre NaN lanza `ValueError`. Añadida guardia `np.isnan(val)` en `_pixel_value`.
- [ ] **`None == None` en `clc_uniform_surroundings`** (`clc_enrichment.py`) — píxeles sin datos se comparan como iguales, marcándolos como "uniformes". Añadir comprobación `df["clc_class"].notna()` a la expresión booleana.
- [ ] **Off-by-one en wrapping de `_resolve_neighbor` para ring 3** (`clc_enrichment.py`) — hardcodea `9999`/`0` que es incorrecto para offsets ±2. Usar aritmética modular: `new_row % 10000`. Bug latente (ring 3 no habilitado).
- [ ] **Nombres de columnas hardcodeados en `clc_uniform_surroundings`** (`clc_enrichment.py`) — los sufijos se calculan dinámicamente pero la comparación hardcodea `N`, `S`, `W`, `E`. Debería usar la lista dinámica.
- [ ] **Dimensión de tile hardcodeada `10000`** (`clc_enrichment.py`) vs `reader.height/width` dinámico en otro lado. Extraer a constante nombrada o pasar como parámetro.
- [x] **Import muerto** (`clc_enrichment.py`) — `import numpy as np` no se usaba. Ahora se usa para la guardia `np.isnan`.
- [ ] **Función muerta** (`clc_enrichment.py`) — `load_clc_classes()` nunca se llama. Usar o prefijar con `_`.
- [x] **Anotación de tipo incorrecta** (`clc_enrichment.py`) — `_pixel_value` ahora devuelve correctamente `-> int | None` (antes devolvía `float` para valores no enteros).
- [x] **Inconsistencia en docstring** (`clc_enrichment.py`) — actualizado para reflejar que solo se añaden columnas `clc_class` (eliminadas columnas `clc_name` del resultado).
- [ ] **Sin `try/finally` para limpieza de readers** (`clc_enrichment.py`) — descriptors de archivo se pierden en excepción. Envolver bucle principal en `try/finally`.
- [ ] **Exports missing** (`enrichment/__init__.py`) — `NEIGHBORHOOD_RINGS`, `CLC_LABELS`, `load_clc_classes` son públicos pero no están en `__all__`. Exportar o prefijar con `_`.
- [x] **Eliminar columnas `clc_name`** — columnas de nombre del píxel central y vecinos eliminadas del resultado de `enrich_with_clc`. Solo permanecen las columnas `clc_class` (int).
- [ ] **Sin tests para enriquecimiento CLC** — cero cobertura para `_resolve_neighbor`, `NEIGHBORHOOD_RINGS`, esquema de salida de `enrich_with_clc`, `clc_uniform_surroundings`.

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
