# Nota metodológica — bloque 03 (aprendizaje no supervisado)

Cuadernos `03a`–`03e`, módulos `src/wildfire/processing/unsupervised.py`,
`events.py`, `typology.py` (con tests), base 23.142 detecciones 2023–2024.
Estado: tipología aceptada (conjunto A, k=12); promoción a modelo pendiente
del contrato de ablación (requiere días de fondo, aún inexistentes).

## 1. Qué responde (y qué no)

Cada detección FIRMS trae las condiciones de su día (calor, viento, sequedad,
vegetación, época). El bloque agrupa las detecciones por **similitud de
condiciones** para responder tres preguntas:

1. ¿Hay *tipos* distinguibles de días con fuego? (tipología de condiciones).
   Describen cómo son los días *con* fuego; si sirven para predecir lo decide
   §7B, no el clustering.
2. ¿Qué detecciones son *raras* respecto al resto? (extremos o artefactos).
3. ¿Cuántos **incendios reales** hay detrás de las detecciones? (un incendio
   se detecta varios días seguidos).

## 2. Datos y filtros decididos

FIRMS España 2023–2024 (MODIS + VIIRS fusionados, `01a`), clima por lotes
Open-Meteo, CLCPlus (pixel + 8 vecinos), presupuestos CCAA por hectárea
MITECO (`01b`). Tres filtros de equipo, en este orden:

1. `clc_class != 1` (sellado) + `type == 0` (vegetación).
2. **Quemas candidatas, 01b-2.11 (reversible)**: se apartan las detecciones
   *dentro* de la ventana 15 oct–30 may (por grupo de CCAA norte/centro/sur,
   hoy idéntica) *y* sobre herbáceas (`clc_class` 6/7). Lógica: en meses fríos
   no hay régimen de incendio forestal sobre herbáceas; esas detecciones son
   casi con seguridad quemas gestionadas. En verano se conserva todo.
   Puerta de calidad: las apartadas deben tener FRP menor (5,21 vs 3,82 ✓).
3. Base resultante: **30.412 → 23.142**; las 7.270 apartadas se conservan en
   `expanded/quemas_agricolas_candidatas.csv` para perfilar aparte.
   Ceuta/Melilla fuera (sin presupuestos comparables). `sup_forestal_ha` se
   conserva en el marco (58 columnas `cleaned`, 61 `lags_index`).

## 3. Conjuntos y reglas duras

- **A**: 7 PCs meteorológicos (12 variables + par circular, `log1p` en
  precipitación/rachas/VPD, z-score). El PCA (k=7 al 91,5 %) decorrelaciona
  los pares colineales (temperatura↔VPD, humedades de suelo, ET0↔radiación)
  que ponderarían doble en distancias euclídeas. PC1 = calor-sequedad;
  `corr(PC1, indice_meteorologico) = 0,919`: se usa uno, nunca ambos.
- **B**: A + estacionalidad (`month_*`, `day_of_year_*` sin escalar) +
  `franja_horaria` ordinal. **C**: B + entorno CLC colapsado (centro nominal
  fuera de distancias + homogeneidad + frecuencias vecinas). **D** (solo 03d):
  lat/lon a km + día continuo.
- Reglas: **nada de satélite en entradas** (`frp`, `brightness`,
  `confidence` solo perfilan/validan); un concepto, una representación (sin
  `rain`, sin mes crudo, sin dirección cruda, sin `is_day`); cada scaler se
  ajusta dentro de cada experimento; semilla 42 en todo.

## 4. Métodos (solo sklearn + scipy)

- **Tipología (03b)**: KMeans k=2–12 (línea base) con inercia, silhouette,
  Calinski-Harabasz, Davies-Bouldin; GMM `full`/`diag` por BIC (pertenencia
  suave — `full` gana a todo k: dentro de cada régimen las nubes son elipses
  orientadas); Ward en muestra de 5.000 (corte independiente, sin salto
  nítido ~67→78); **controles negativos** mes (12) y CCAA (17) que la
  candidata debe batir en severidad.
- **Anomalías (03c)**: Isolation Forest, LOF, Mahalanobis robusta sobre A/B;
  bandera = 2 % superior de cada score, consenso ≥2 detectores (A: 249 filas,
  1,08 %; B: 171, 0,74 %). IForest∩Mahalanobis acuerdan (1,0 %), LOF caza
  aparte (0,1 %). Contrastes: 22,9 % del consenso en top-15 % FRP (≥25,6),
  4,8 % sobre BLH imputada, top Canarias/Asturias/Andalucía, meses 8/3/9.
- **Eventos (03d)**: DBSCAN sobre (x_km, y_km, día×7,5) en **EPSG:3035**
  (cubre Canarias; UTM 30N no) con eps=10 (rodilla aproximada, tramo estable
  5–20) y `min_samples=3`: **1.177 eventos, 21,4 % ruido** (bajo el listón
  de replanteo del 30 %). El 29 % de eventos atraviesa >1 régimen; la carga
  de anomalías no crece con la duración.

## 5. Validación 7A ∧ 7B (03e) y veredicto

Candidatos: A con k ∈ {5, 7, 12} (único conjunto que supera al control CCAA
con NMI bajo; B *es* el calendario con NMI-mes ~0,5; C rara vez supera los
controles). Escalas de ARI invariantes a reetiquetado (emparejamiento
húngaro donde haría falta); la primera métrica temporal ingenua se detectó
y corrigió antes del veredicto.

| Bloque | k5 | k7 | k12 |
|---|---|---|---|
| 7A remuestreo 30×80 % (bar 0,7) | 0,465 ✗ | 0,709 ✓ | 0,946 ✓ |
| 7A espacial (sobreviven los k grupos) | ✓ | ✓ | ✓ |
| 7A temporal 2023↔2024 (listón de juicio 0,5) | 0,471 ✗ | 0,570 ✓ | 0,683 ✓ |
| 7A preprocesado Std/Rob (bar 0,6) | 0,450 ✗ | 0,440 ✗ | 0,605 ✓ |
| 7A sentido físico (VPD↑, humedad↓, ET0↑) | ✓ | ✓ | ✓ |
| 7B severidad max\|δ\| (controles 0,401/0,699) | 0,714 ✓ | 0,847 ✓ | 0,863 ✓ |
| 7B no-coincidencia / no-trivialidad / solape GMM | ✓ | ✓ | ✓ |

**Veredicto: A/k12 aceptada; k5 y k7 rechazadas.** Sin 7A se rechaza; con 7A
pero sin 7B quedaría no concluyente. Hallazgos de lectura: k5/k7 mueren en
robustez de preprocesado (el scaler mueve sus fronteras — fragilidad real de
las colas pesadas, no artefacto métrico); Canarias es sistemáticamente la
peor región en replicación (régimen insular distinto, coherente con 03c);
`min_cliff_delta` quedó **anclado en 0,70** (mejor control CCAA 0,699 +
paso de 0,05), en `configs/project.yaml`. Tabla completa en
`unsupervised/validacion_resumen.csv`; perfiles finales en
`unsupervised/perfiles_tipologia.csv`.

## 6. Límites honestos

- **No hay probabilidad de incendio**: solo existen días *con* detecciones;
  sin días de fondo no se puede predecir cuándo habrá fuego.
- **La separación quema/incendio es heurística**: la 2.11 aparta quemas
  probables por temporada + combustible, con puerta FRP y archivo separado —
  pero Asturias/Cantabria arden en invierno de verdad, y parte del ruido de
  03d cae en herbáceas estivales. El filtro ayuda, no certifica.
- **Dos años no son clima**: todo describe 2023–2024; temperatura, presupuestos
  y ventanas son de ese bienio.
- **Estable ≠ útil, bonito ≠ evidencia**: la aceptación exige 7A ∧ 7B; la
  ablación ΔPR-AUC con `cluster_meteo`/`score_anomalia` queda pendiente de
  los días de fondo (columnas declaradas no probadas hasta entonces).
- **Canarias es otro régimen**: peor replicación espacial/temporal y top en
  anomalías; cualquier uso estatal de la tipología debe tratarla aparte.
- **12 grupos son muchos**: k12 gana por estabilidad y severidad, no por
  parsimonia; la interpretación grupo a grupo vive en `perfiles_tipologia.csv`.

## 7. Salidas y trazabilidad

`unsupervised/`: `features_{a,b,c}.csv` + `perfilado.csv` (03a),
`clustering_*`, `cluster_labels.csv` (k5 + probas GMM), `severidad/nmi/
perfiles_kmeans_*`, `ward_linkage_a.csv` (03b), `anomalias_{a,b}.csv` +
`contrastes_anomalias.csv` (03c), `eventos.csv` + `eventos_resumen.csv`
(03d), `validacion_resumen.csv` + `perfiles_tipologia.csv` (03e). Clave `detection_id`
(nace en 02a como orden de filas, primera columna de `lags_index.csv` y de
todas las salidas 03a): todo join del bloque es por clave verificada
(`check_ids`), nunca por posición. Cuadernos `03a`–`03e` en español (orquestan;
la lógica >5 líneas vive en `src/` con tests: `test_unsupervised.py`,
`test_events.py`, `test_typology.py`).
