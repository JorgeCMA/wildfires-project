# Guía de la demo (Streamlit + Folium)

Explicación completa de qué hace la demo, qué muestra cada pestaña, cada selector y cada vista.

- Entry point: `demo/app.py`
- Extra opcional: `pip install -e ".[demo]"` (`streamlit`, `folium`, `streamlit-folium`)
- Arranque, desde la raíz del proyecto:

  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run demo/app.py
  ```

- Documentos relacionados: `demo/README.md`, `docs/deployment-unit-A-B.md`, `docs/calibration-C.md`, `docs/background-pool-status.md`.

---

## 0. Idea general en un párrafo

La demo es **de solo lectura salvo por una acción**. Cuatro de las cinco pestañas
se limitan a mostrar artefactos ya escritos (CSV/GeoJSON) por los notebooks 01/04
y por `scripts/fit_occurrence_model.py`; lo único que calcula algo al pulsar un
botón es el **scorer en vivo de Plan C**. Nada de lo que hay en el mapa es salida
del modelo: el mapa muestra **dónde ardieron de verdad** (detecciones FIRMS). Las
cifras derivadas del modelo (tramos, tasas base, probabilidades de excedencia)
están calculadas sobre el **pool de fondo completo** (congelado el 2026‑10‑25).

### Artefactos que lee

| Artefacto | Alimenta |
|---|---|
| `expanded/firms_spain_final_lags_index.csv` (23.142 detecciones, 2023‑01‑01 → 2024‑12‑31, 17 CCAA) | Opciones de fecha/CCAA de la barra lateral, **Explorar mapa** |
| `predictions/occurrence_scores.csv` (138.852 filas = 2 protocolos × 69.426) | **Plan A** |
| `predictions/ccaa_day_fire_rates.csv` (17 CCAA + `ALL`) | **Plan B** (tabla + coropleta) |
| `raw/ccaa/spain-communities.geojson` | Geometría de Plan B |
| `predictions/calibration_reliability.csv` (3 márgenes × 10 bins) | Curvas de **Plan C** |
| `predictions/occurrence_fold_metrics.csv` (125 filas) | **Estado** |
| `models/occurrence_bundle.joblib` | **Scorer en vivo de Plan C** (vía `scripts/fit_occurrence_model.py`) |

Si falta un fichero, el cargador lanza `FileNotFoundError` y `_safe(...)` muestra
un mensaje rojo *"Falta un artefacto…"* indicando qué notebook hay que ejecutar,
en lugar de un traceback.

---

## 1. Estructura de la página

- **Título**: *"Wildfire occurrence — demo (Plan D)"*.
- **Banner** (siempre visible): *"Fondo de negativos completo. N filas de fondo
  (ratio objetivo ≥ 2)…"*. La `N` sale en vivo de
  `pool_status()['pool_rows']` (hoy 97.630) y el pool está congelado desde el
  2026‑10‑25.
- **Layout**: ancho (`layout="wide"`), barra lateral a la izquierda y 5 pestañas
  arriba.

---

## 2. Barra lateral — "Controles"

Hay cuatro controles y cada uno afecta a una **pestaña concreta**:

1. **`Protocolo de validación`** — radio: `season` / `ccaa`.
   Elige qué protocolo de validación bloqueada (*leave-one-group-out*) se muestra
   en **Plan A**. Los dos protocolos son particiones por trimestre-año vs. por
   comunidad; como las puntuaciones de pliegues distintos no comparten escala, los
   tramos se calculan *dentro* de cada protocolo. Hoy: 69.426 filas puntuadas por
   protocolo.

2. **`Mapa (detecciones)` → `Rango de fechas`** — rango de fechas (por defecto el
   span completo 2023‑01‑01 → 2024‑12‑31; mín/máx salen de los datos).
   Filtra la capa de puntos de **Explorar mapa**. (Si seleccionas una sola fecha
   se ignora: el código solo aplica el rango cuando hay dos extremos.)

3. **`Comunidades`** — multiselección de las 17 CCAA (por defecto vacío = todas).
   También filtra solo la capa de **Explorar mapa**.

4. Un `caption` que recuerda que el cálculo en vivo está en **Plan C**.

Ojo: los dos filtros del mapa **no** afectan a Plan A/B/C, y el radio de protocolo
no afecta a B/C.

---

## 3. **Explorar mapa** — detecciones crudas en el tiempo

Objetivo: *dónde y cuándo hubo detecciones*, sin modelo.

- Aplica los filtros de fecha y CCAA de la barra lateral y muestra:
  - **Radio `Capa`** — `Clúster` o `Mapa de calor`. Los mismos puntos, dos
    renderizados:
    - `Clúster` (`FastMarkerCluster`): los puntos se agrupan en burbujas con
      recuento y, al acercarte, cada detección se dibuja como una **llama de
      fuego** (Font Awesome `fa-fire` sobre marcador rojo). Es un `callback`
      propio: el marcador por defecto de Folium es una casita
      `glyphicon-home`. El callback debe ser la *expresión* de la función
      (Folium lo envuelve en `var callback = …;`); incluir `var callback =`
      dentro del string rompe el script del mapa.
    - `Mapa de calor` (`HeatMap`, radio 6, blur 8): solo el gradiente, sin iconos.
  - Un `caption`: *"N detecciones mostradas."* (el recuento filtrado).
  - Un **mapa Folium** sobre España (tiles OpenStreetMap, zoom 6).
  - Debajo, un **gráfico de barras mensual** (`st.bar_chart`) de detecciones por
    `YYYY-MM` del subconjunto filtrado: la vista temporal.

Es la capa "verdad terreno" honesta: si no seleccionas nada más, esto es
simplemente el registro FIRMS.

---

## 4. **Plan A — tramos de riesgo relativos**

Objetivo: el **ranking** del modelo, troceado en tramos de triaje relativos.

- Un `caption` fija la advertencia clave: los tramos son **relativos al lote
  puntuado**, no una escala absoluta de peligro; modelo base: logístico.
- **Gráfico de barras** `Distribución por tramo (relativo)`: recuento de filas
  puntuadas por tramo, coloreado de verde a rojo con `RISK_BAND_COLORS`. Como los
  tramos son **quintiles por protocolo**, con 69.426 filas las barras salen
  ~iguales por construcción (13.885 en cuatro tramos y 13.886 en *Muy alto*). El
  gráfico sobre todo confirma el troceado, no la habilidad.
- **Tabla** `score_medio` por CCAA: puntuación media por región, descendente. Aquí
  es donde Plan A dice algo útil: qué regiones rankean más alto **bajo el
  protocolo actual**.

Columnas detrás: `record_id, label, score, protocol, held_out, risk_band,
risk_percentile, ccaa, ccaa_fire_rate`.

### 4.1. Qué cambia al pasar de `season` a `ccaa`

Las dos opciones usan **las mismas 69.426 filas puntuadas y las mismas columnas**;
lo único que cambia son las predicciones *out-of-fold* según **cómo se bloquea la
validación**:

| Protocolo | Bloqueo (*leave-one-group-out*) | Pliegues (`held_out`) | Qué prueba |
|---|---|---|---|
| `season` | por **trimestre-año** | `2023-Q1 … 2024-Q4` (8) | generalizar **en el tiempo**, con las mismas regiones en entrenamiento |
| `ccaa` | por **comunidad** | las 17 CCAA (una por pliegue) | generalizar a una **región no vista** en entrenamiento |

Dicho con precisión, qué queda **fuera** y qué **dentro** en **cada pliegue
individual** (no en el conjunto de todos los pliegues):

| Selección | Fuera del entrenamiento (test) | Dentro del entrenamiento |
|---|---|---|
| `season` | **un** trimestre-año entero (p. ej. `2023-Q1`): sus positivos **y** su fondo | los otros 7 trimestres, con **todas** las CCAA |
| `ccaa` | **una** CCAA entera (p. ej. `Galicia`): sus positivos **y** su fondo | las otras 16 CCAA, con **todos** los trimestres |

En resumen: al elegir `season` **se queda fuera la season** (una por pliegue) y las
CCAA se quedan dentro; **no** es "seasons dentro + CCAA fuera". La opción que deja
fuera las CCAA es `ccaa`. En ambos casos el fondo (negativos) se bloquea con la
misma columna que los positivos, así que no se cuela *leakage* por el otro eje.

En ambos casos las puntuaciones son las del **modelo base logístico** (predicción
fuera de pliegue). Consecuencias al pulsar uno u otro:

1. **El gráfico de barras (`Distribución por tramo`) NO cambia.** Los tramos son
   quintiles calculados *dentro* de cada protocolo sobre el mismo número de filas,
   así que los recuentos son idénticos en los dos: Muy bajo 13.885, Bajo 13.885,
   Medio 13.885, Alto 13.885, Muy alto 13.886. El gráfico no distingue protocolos.

2. **La escala de las puntuaciones sí se desplaza.** El score medio pasa de
   ≈ 0,30 (`season`) a ≈ 0,39 (`ccaa`); el rango va de ~0 a ~0,999 en ambos.
   No es un cambio de calidad, es el efecto de bloquear por región.

3. **La tabla `score_medio` es lo que cambia de verdad**: cambian los valores por
   CCAA y, sobre todo, **el orden de las regiones**. La correlación de rangos entre
   los dos protocolos es solo **≈ 0,52 (Spearman)**: la mitad del ranking se
   reordena. Mayores movimientos:

   | CCAA | Rank `season` | Rank `ccaa` | Salto |
   |---|---:|---:|---:|
   | Galicia | 9 | 1 | ↑ 8 |
   | La Rioja | 17 | 10 | ↑ 7 |
   | Comunidad Valenciana | 4 | 14 | ↓ 10 |
   | Cataluña | 8 | 15 | ↓ 7 |
   | Región de Murcia | 5 | 11 | ↓ 6 |

   Por ejemplo, bajo `season` Galicia queda 9.ª y la Comunidad Valenciana 4.ª;
   bajo `ccaa` Galicia pasa a 1.ª y la Comunidad Valenciana cae a 14.ª.

**Por qué.** Con `season` la región está en el entrenamiento (solo se aparta un
trimestre), así que el modelo puede apoyarse en el efecto propio de cada región.
Con `ccaa` la región entera queda fuera, y el modelo tiene que decidir con
features no regionales: las predicciones (y con ellas el ranking) se mueven.

**Advertencia de lectura**: ninguno de los dos protocolos es "la verdad". El orden
de regiones **depende del protocolo**, así que el `score_medio` de Plan A debe
leerse como *"bajo este protocolo"*, nunca como un ranking absoluto de peligro
(que es justo lo que dice el `caption` de la pestaña).

---

## 5. **Plan B — tasa base empírica CCAA × día**

Objetivo: la tasa base de evento raro **medida y sin modelo**.

- `caption`: `P(≥1 detección en la CCAA ese día)`.
- **Coropleta** (`folium.Choropleth`, `YlOrRd`): cada CCAA sombreada por
  `fire_rate`, con clave `feature.properties.ccaa_es`. Se excluye la fila agregada
  `ALL`; las regiones sin tasa (Ceuta/Melilla, o nombres no mapeados) salen en gris
  (`nan_fill_color`). La geometría se limpia (`_to_multipolygon`) para que la
  geometría insular no rompa los límites de Folium.
- **Tabla**: las 18 filas, con `fire_rate` en porcentaje.

`fire_rate` = (nº de días con ≥1 detección en esa CCAA) ÷ (nº de días del span
2023‑01‑01→2024‑12‑31, 731). Instantáneas: Andalucía ≈ 66,8 %, Castilla‑La Mancha
≈ 49,7 %, Extremadura ≈ 45,6 %, … y la agregada `ALL` ≈ 26,8 %. Es el "cada cuánto
tiene fuego esta región en un día cualquiera" que un plan puede usar — sin modelo
y sin cuota.

---

## 6. **Plan C — excedencia**

Dos secciones apiladas.

### 6a. Curvas de fiabilidad (precalculadas)

- `Fiabilidad por margen`: probabilidad predicha (x) vs frecuencia observada (y)
  por margen, una línea por `margin_km` ∈ {5, 10, 20}, más la diagonal ideal.
  Datos = `calibration_reliability.csv` (10 bins por cuantiles × 3 márgenes).
- **Tabla** de esas mismas 30 filas: `bin_lower, bin_upper, mean_predicted,
  observed_freq, n, margin_km`.
- El `caption` recuerda que es una probabilidad de **margen ancho** (no puntual):
  calibración isotónica entrenada en 2023, calibrada en 2024, corregida por la
  tasa base medida.

### 6b. `Scorer en vivo — un punto, una fecha` (la única acción en vivo)

**Mapa selector (clic).** Lo primero es un mapa Folium con un marcador en el punto
activo: al **hacer clic** en cualquier parte del mapa, esa coordenada pasa a los
campos `Latitud`/`Longitud` (redondeada a **4 decimales** y recortada a los límites
27–44,5° N / 18,5° O–4,5° E). El punto se guarda en `st.session_state`
(`pc_lat`/`pc_lon`) y es el que comparten el mapa y el formulario; también puedes
escribir las coordenadas a mano. El mapa usa `returned_objects=["last_clicked"]`
(solo devuelve el clic, así no provoca reruns al hacer zoom o desplazar).

Formulario con:

| Control | Por defecto | Significado |
|---|---|---|
| `Latitud` | 40.4168 | latitud del punto (WGS84), 4 decimales |
| `Longitud` | -3.7038 | longitud del punto, 4 decimales |
| `Fecha` | 2024‑07‑15 | día (rango 2020‑01‑01 → 2025‑12‑31) |
| `Hora (UTC)` | 12 | slider 0–23, se usa para interpolar el clima horario |
| `Margen` | 10 | 5/10/20 km — qué margen destaca la métrica principal |
| **`Calcular excedencia`** | — | el único botón que hace trabajo |

Al pulsarlo:

1. `_bundle()` carga `models/occurrence_bundle.joblib` (cacheado en memoria con
   `st.cache_resource`; si falta → mensaje rojo que te dice que ejecutes el script
   de ajuste).
2. `build_point_features(lat, lon, date, hour)` descarga **clima de Open-Meteo en
   vivo** para ese punto/día, lo interpola a la hora pedida y encadena CLC → CCAA →
   sumas de presupuesto → features de ingeniería, terminando exactamente en el
   contrato `EVERYDAY_FEATURES` del modelo. **Este es el único punto donde la demo
   gasta cuota de API** (una petición por pulsación).
3. `score_point(...)` produce el score bruto + la excedencia calibrada por margen.

Salidas mostradas (el punto sigue marcado en el mapa de arriba):

- Una **métrica**: `P(≥1 detección a ≤ R km ese día)` para el margen elegido, en %.
- Un `caption` con el **score bruto del modelo**.
- Una **tabla de 3 filas** (`margen_km`, `probabilidad` %) para ver los tres
  márgenes, no solo el elegido.

Cualquier fallo (red, tile CLC ausente, contrato de features) se captura y se
muestra como mensaje rojo en lugar de tumbar la app.

---

## 7. **Estado** — panel del fondo

Tres elementos:

1. Un **bloque monoespaciado** de estado (`format_pool_status(pool_status())`):
   relleno del parcial de clima (`with weather` / `fully complete` / `resume row`),
   el recuento de filas del pool enriquecido, una tabla por año
   (`positives / needed / have / ratio`) y una línea **VERDICT** — hoy
   `READY - every year reaches the ratio target; pool complete (frozen 2026-10-25)`.
   Responde a "¿ya está congelado el pool?" (objetivo = `background_ratio: 2`
   negativos por positivo **por año**).
2. Una **lista de advertencias** en markdown: el mapa son detecciones crudas, no el
   modelo; los tramos son relativos; la excedencia es de margen ancho (no puntual);
   el PR‑AUC está inflado por el enmarcado ~1:2 (es un ranking, no precisión de
   campo); enlace a `docs/background-pool-status.md`.
3. Una **tabla de métricas por pliegue** (`occurrence_fold_metrics.csv`, 125 filas):
   `protocol, held_out, n_train, n_test, prevalence, pr_auc, roc_auc, model`.
   Contiene los 5 modelos candidatos (`logistic`, `logistic_balanced`,
   `random_forest`, `random_forest_balanced`, `hist_gradient_boosting`) en ambos
   protocolos y todos los pliegues, para ver la habilidad por pliegue — incluido el
   pliegue casi perfecto `season/2024-Q1` que el notebook 04 audita aparte.

   **Cómo leerla**: `held_out` es el grupo **excluido del entrenamiento** (el test)
   y `n_test` es su tamaño; `n_train` es **todo lo demás** del marco de modelado
   (69.426 filas = 23.142 detecciones + 46.284 de fondo, ratio 2), por eso sale
   ~58k–69k en todos los pliegues. Para ver cuánto dato aporta cada season/CCAA hay
   que mirar **`n_test`** (495–11.368), no `n_train`. Siempre
   `n_train + n_test = 69.426`, y cada `held_out` aparece **5 veces** (una por
   modelo candidato). `prevalence` es la mezcla de etiquetas del pliegue de test en
   el marco **muestreado** (~1:2), no la tasa real de fuego.

---

## 8. Qué entrenamiento hay detrás de cada pestaña

**La demo no entrena nada al usarse**: `demo/app.py` no contiene ninguna llamada a
`.fit()` (ni al arrancar ni al pulsar botones). Todo el entrenamiento ocurrió antes,
en los notebooks 01/04 y en `scripts/fit_occurrence_model.py`; la demo solo lee
artefactos y, como mucho, **aplica** un modelo ya entrenado.

| Sección | ¿Entrena? | Qué hay detrás | Dónde se entrenó |
|---|---|---|---|
| **Explorar mapa** | No | Nada: detecciones FIRMS crudas con filtros de fecha/CCAA | — |
| **Plan A — tramos** | No en la app (sí aguas arriba) | Un **logístico por pliegue** de la validación bloqueada (8 `season` / 17 `ccaa`); puntuaciones **out-of-fold**; los tramos son quintiles de esas puntuaciones, por protocolo | `notebooks/04` |
| **Plan B — tasa base** | **No** (ni aquí ni aguas arriba) | Frecuencia empírica `días con ≥1 detección / días del span`; sin modelo | `notebooks/04` (solo conteo) |
| **Plan C — curvas** | No en la app (sí aguas arriba) | HGB entrenado en **2023** + **isotónica por margen** ajustada en **2024** (cross-fit por bloques de días) + corrección de prior | `notebooks/04` §8 |
| **Plan C — scorer en vivo** | **No** | Inferencia pura con el bundle: preprocesado + HGB(2023) + isotónicas(2024) | `scripts/fit_occurrence_model.py` |
| **Estado** | **No** | Recuentos del pool y métricas ya calculadas de la CV | `notebooks/04` y `scripts/background_pool_status.py` |

### 8.1. Plan A — un modelo por pliegue (evaluación, no despliegue)

- **Estimador**: `model_candidates(...)["logistic"]`, es decir
  `LogisticRegression(max_iter=1000, random_state=42)`.
- **Cuántos ajustes**: uno por pliegue, con `clone(modelo).fit(X_train, y_train)`:
  **8** con `season` y **17** con `ccaa` (25 en total entre los dos protocolos).
- **Preprocesado por pliegue** (`prepare_fold`, ajustado **solo con el train del
  pliegue**, nunca con el test): one-hot de las categóricas con las columnas del
  train, imputación por **medianas del train** y `StandardScaler` sobre las
  continuas del train. El test se transforma con esos objetos.
- **Out-of-fold**: la puntuación de cada fila viene del pliegue en el que esa fila
  quedó fuera, así que ninguna fila se puntúa con un modelo que la vio.
- **Sin calibración**: son puntuaciones de *ranking*, no probabilidades (su nivel
  lo fija el muestreo de fondo ~1:2).
- **Tramos**: `add_risk_bands(..., group_col="protocol", n_bands=5)` calcula
  quintiles **por protocolo** sobre las 69.426 filas puntuadas (positivos **y**
  fondo muestreado, no solo las 23.142 detecciones). Por eso los recuentos son
  casi idénticos entre protocolos y el gráfico de barras apenas informa.
- **La app**: `load_scores(protocol)` + `band_counts` + media por CCAA. No ajusta
  nada.

### 8.2. Plan B — nada de entrenamiento (conteo puro)

- `ccaa_day_fire_rates(positives)` solo cuenta: nº de `(CCAA, día)` con ≥1
  detección dividido por los **731 días** del span 2023‑01‑01 → 2024‑12‑31 (el
  mismo denominador para cada CCAA), más una fila agregada `ALL`.
- **No hay modelo, ni split, ni calibración.** Es una tasa base descriptiva: no
  corrige estacionalidad ni meteorología; sirve de referencia operativa junto al
  ranking.

### 8.3. Plan C — curvas: dos ajustes (modelo + isotónica) y corrección de prior

Sí hay entrenamiento aguas arriba: **dos ajustes** sobre una **partición temporal
única** (no CV del estimador), más una corrección de prior:

1. **Modelo** — `HistGradientBoostingClassifier(random_state=42)` (hiperparámetros
   por defecto de scikit-learn), ajustado **una vez** sobre 2023: 44.082 filas =
   14.694 positivos + 29.388 de fondo (ratio 2).
2. **Calibrador isotónico por margen** — para cada `R ∈ {5, 10, 20}` km se etiqueta
   `y = 1` si la fila de validación tiene una detección a ≤ R km **el mismo día**
   (una detección está a 0 km de sí misma, así que las filas positivas salen
   siempre `y = 1`; el margen separa sobre todo las filas de fondo), y
   se ajusta `IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)` sobre
   2024 (25.344 filas = 8.448 positivos + 16.896 de fondo).
   - **Cross-fit**: los días únicos se reparten en 5 bloques; la probabilidad de
     cada bloque sale de una isotónica ajustada con los otros 4. Así las curvas de
     fiabilidad son **out-of-fold** (no se puntúan con su propio ajuste).
   - El **calibrador de despliegue** (el que guarda el bundle) se reajusta con
     **todo 2024**.
3. **Corrección de prior** — `prior_correct` reescala las odds desde la prevalencia
   de la muestra de 2024 (≈33,4 / 33,6 / 34,5 %) a la **tasa base medida en el
   pool de fondo**: 0,123 / 0,464 / 1,710 % para 5/10/20 km. La media de las
   probabilidades corregidas es ≈0,23 / 0,85 / 3,00 %; Brier ≈0,15 y
   ECE ≈0,023–0,027.

La curva que ves (`mean_predicted` vs `observed_freq`) evalúa las predicciones
**cross-fitted**, no el calibrador final; y la app solo lee el CSV de 30 filas.

### 8.4. Plan C — scorer en vivo: cero entrenamiento

- El bundle lo construye `scripts/fit_occurrence_model.py` con la **misma ruta C**
  (HGB 2023 + isotónicas 2024) y guarda: preprocesado (columnas one-hot, medianas,
  scaler, orden de features), el estimador y un `MarginCalibrator` por margen
  (isotónica final + prevalencias origen/destino).
- Al pulsar **`Calcular excedencia`** no se ajusta nada: `build_point_features`
  (Open‑Meteo → interpolación horaria → CLC → CCAA → presupuestos → ingeniería)
  → `transform` → `predict_proba` → isotónica → corrección de prior. Solo se gasta
  **una petición HTTP** por pulsación. El punto sale del mapa selector (clic) o de
  los campos `Latitud`/`Longitud` (comparten `st.session_state`).
- `st.cache_resource` mantiene el bundle en memoria: pulsar otra vez **no
  reentrena**; solo recalcula features e inferencia.
- Para regenerarlo: `python scripts/fit_occurrence_model.py` (artefacto
  reproducible y *gitignored*).

### 8.5. Estado — lectura de resultados ya calculados

- No entrena: lee los recuentos del parcial de clima y del pool (`pool_status`) y
  el CSV de métricas por pliegue.
- Ese CSV tiene 125 filas = (8 pliegues `season` + 17 `ccaa`) × 5 modelos
  candidatos: es la salida de la misma CV bloqueada que Plan A, pero para todos los
  candidatos (`logistic`, `logistic_balanced`, `random_forest`,
  `random_forest_balanced`, `hist_gradient_boosting`).

### 8.6. Resumen de esquemas

| Sección | Esquema | Para qué |
|---|---|---|
| Plan A | *leave-one-group-out* (`season` o `ccaa`) | **evaluación** de ranking; no es el modelo desplegado |
| Plan C | split temporal **2023 → 2024** | modelo en 2023, calibración en 2024; cross-fit solo dentro de 2024 para medir fiabilidad |
| Plan B | ninguno | estadística descriptiva |
| Scorer en vivo | ninguno (modelo congelado) | inferencia |

Ninguna pestaña usa un split aleatorio.

---

## 9. Glosario de lo que significan los números

- **`score`** — salida de ranking del modelo; *no* es una probabilidad (su nivel lo
  fija el ratio de muestreo de negativos).
- **`risk_band` / `risk_percentile`** — tramo por quintiles y percentil de orden
  **dentro del lote puntuado de un protocolo**; relativos, no peligro absoluto.
- **`ccaa_fire_rate`** — la tasa base empírica por región que también muestra
  Plan B.
- **`fire_rate`** — `P(≥1 detección | CCAA, día)`, medida, sin modelo.
- **Probabilidad de excedencia** — `P(≥1 detección a ≤ R km ese día)` calibrada;
  solo tiene sentido a márgenes anchos (5/10/20 km); isotónica + corrección de
  prior.
- **Pool** — el pool de negativos está **completo y congelado** (2026‑10‑25,
  97.630 filas, ratio ≥ 2 en ambos años), así que los scores y las cifras de
  calibración son finales para esa instantánea.

---

## 10. Lo que deliberadamente **no** es

- No es una probabilidad de fuego a nivel de punto (solo excedencia de margen ancho).
- No es un pronóstico: el scorer en vivo descarga clima **histórico** para la fecha
  elegida; "en vivo" significa calculado a petición, no predicción futura.
- No está congelada a nivel de **método**: la excedencia es una aproximación de
  margen ancho y el modelo es un ranking. El **pool de fondo sí está congelado**
  (25‑10‑2026), así que las cifras actuales son finales para esa instantánea.

---

## 11. Usarla en 30 segundos

1. Barra lateral: elige protocolo + rango de fechas + regiones.
2. **Explorar mapa**: alterna Clúster / Mapa de calor; mira las barras mensuales.
3. **Plan A**: lee la tabla de score medio por CCAA.
4. **Plan B**: lee la coropleta de tasa base (Andalucía/CLM/Extremadura arriba).
5. **Plan C**: mira las curvas de fiabilidad y luego introduce un punto/fecha/hora y
   pulsa **Calcular excedencia** para obtener la excedencia calibrada en los tres
   márgenes (gasta una petición a Open-Meteo).
6. **Estado**: confirma el `VERDICT: READY` (pool congelado) e inspecciona el
   PR‑AUC por pliegue.
