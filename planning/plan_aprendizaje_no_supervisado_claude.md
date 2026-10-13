# Plan de aprendizaje no supervisado — Incendios en España (FIRMS 2023–2024)

Basado en las salidas de los cuadernos 01a, 01b, 02a, 02b y 02c. Entrada principal: `data/processed/expanded/firms_spain_final_lags_index.csv` (y su copia `firms_spain_final_normalized.csv`).

---

## 1. Punto de partida (lo que muestran los resultados)

| Hecho observado | Fuente | Consecuencia para el plan |
|---|---|---|
| 30.412 filas × 60 columnas tras filtrar `clc_class != 1`, `type == 0`, quitar Ceuta/Melilla y deduplicar sensores | 01a / 01b | Tamaño cómodo: se pueden usar métodos con matriz de distancias (Gower, HDBSCAN, GMM) sin muestrear |
| Solo hay **detecciones de incendio**, no días sin fuego | 01a, aviso de 02a | Los clusters describen *tipos de condiciones de incendio*, no riesgo. No se puede inferir "probabilidad de incendio" |
| `confidence_cat`: n 85 %, h 10 %, l 5 % | 02c 6.1 | Útil como variable de **validación externa**, no como entrada |
| PCA meteorológico: 14 variables → 7 PCs al 90,9 % (PC1 34,8 %, PC2 14,5 %, PC3 12,9 %) | 02b 4.1 | PC1 = "sequedad/calor" (ET0, VPD, T, HR−, radiación), PC2 = viento, PC3 = humedad del suelo, PC4 = presión/precipitación, PC5–7 = dirección del viento |
| `corr(PC1, indice_meteorologico)` = 0,916 | 02b | El índice compuesto es casi redundante con PC1: usar uno u otro, no ambos |
| PCA de CLC one-hot: 91 variables → 29 PCs para 95 % (17 para 90 %); PC1–PC5 con cargas casi idénticas en `clc_class` y los 4 vecinos cardinales | 02c 7.2–7.4 | El one-hot + PCA comprime mal variables nominales y mezcla el píxel central con los vecinos. Mejor otra representación (ver §3.3) |
| Kruskal-Wallis: la clase CLC afecta a FRP (H = 1503,7; p ≈ 0) | 02b 5.4 | El uso del suelo contiene señal real y justifica incluirlo en el clustering |
| MLP supervisado: accuracy 0,996 / balanced 0,985, sin convergencia (16 iteraciones) | 02c 6.4 | Resultado **no creíble**: hay fuga de información (ver §2). No sirve como referencia |
| Rezagos (02a): solo existen para fechas con observaciones regionales consecutivas → NaN muy altos; no se exportan | 02a | Excluirlos del clustering salvo que se reconstruyan con clima diario completo |

---

## 2. Problemas a corregir antes de modelar

1. **Fuga / variables de satélite.** El MLP incluyó `confidence_num`, `brightness`, `brightness_ir`, `scan`, `frp`. Para el no supervisado, `brightness`, `brightness_ir`, `scan` y `confidence_num` quedan **fuera de las entradas** (son del sensor, no de la causa). `frp` se reserva como variable de *perfilado* (virulencia), no como entrada, para que los clusters sean "condiciones" y luego se vea si separan virulencia. (Variante opcional con `frp` dentro en §4.5).
2. **`ratio_prevencion_extincion` es constante.** En 01b 2.10 se calcula `prev / (prev + 1e-5)`, que da ≈ 0,999998 en todas las filas (visible en la salida). Debe ser `prevención / (extinción + ε)`. Rehacer antes de usarla; mientras tanto, excluirla.
3. **Presupuestos = constantes por CCAA.** `prevention_current`, `prevention_prev3`, `extinction_prev3` y sus ratios por hectárea solo cambian entre CCAA (17 valores). En un clustering dominarían y producirían clusters = regiones. Se tratan como variables de **perfilado**, no de entrada (o se prueban en una variante separada).
4. **Duplicados de información.** `rain` ≈ `precipitation` (medias 0,017967 vs 0,017980); `year`, `month`/`day_of_year` y sus seno/coseno; `is_day` vs `daynight` vs `franja_horaria`; `indice_riesgo_meteo`, `indice_riesgo_meteo_log`, `indice_meteorologico`, `potencial_propagacion` y `estres_termico_vegetacion` son combinaciones de las mismas variables base. Se elige una sola representación por concepto.
5. **Imputación de `boundary_layer_height`.** 4.928 NaN (16 %) rellenados con la mediana global (la media baja de 822,8 a 783,1 y la desviación de 734 a 678). Esto crea un pico artificial. Alternativas: excluir la variable (PC1 ya la capta con carga 0,69), o imputar por CCAA × mes. Probar ambas.
6. **Sesgo de muestreo.** Asturias (5.339), Andalucía, Extremadura y Canarias dominan; La Rioja tiene 105 y País Vasco 144. Los clusters reflejarán donde hay más detecciones (muchas de ellas quemas agrícolas/prescritas en Asturias/Cantabria en invierno). Se vigila en el perfilado y se prueba una variante con pesos por CCAA o submuestreo.
7. **Escalado.** Usar la versión z-score solo para variables continuas asimétricas **tras transformar**: `log1p` en `precipitation`, `wind_gusts_10m`, `vapour_pressure_deficit`, `boundary_layer_height` (si se mantiene), y quitar `rain`. Los atípicos IQR (precipitación 1.545, VPD 1.760, ráfagas 397…) pesan mucho en k-means; usar `RobustScaler` como alternativa y comparar.
8. **Dos años y pocos eventos extremos.** Cualquier conclusión es descriptiva de 2023–2024; no generalizar a otras temporadas.

---

## 3. Conjuntos de variables (feature sets)

### 3.1 Conjunto A — Meteorología (numérico, el más limpio)
Entrada: PCs 1–7 del PCA meteorológico (90,9 % de varianza), o las 12 variables base escaladas:
`temperature_2m, relative_humidity_2m, wind_speed_10m, wind_gusts_10m, precipitation(log1p), vapour_pressure_deficit(log1p), soil_moisture_0_to_7cm, soil_moisture_7_to_28cm, surface_pressure, shortwave_radiation, et0_fao_evapotranspiration, wd_sin, wd_cos`.
Recomendado: **PCs** (decorrelaciona y reduce el peso de las 5 variables casi colineales de PC1).

### 3.2 Conjunto B — Meteorología + estacionalidad + hora
A + `month_sin, month_cos, day_of_year_sin, day_of_year_cos` + `franja_horaria`/`is_day`.
Objetivo: separar "incendio de verano de tarde" de "quema invernal nocturna".

### 3.3 Conjunto C — Meteorología + uso del suelo
B + descripción del suelo en lugar de one-hot+PCA:
- `clc_class` (píxel central) como categórica.
- Composición del entorno: proporción de los 8 vecinos que comparten clase con el centro, número de clases distintas entre vecinos, y proporción de cada clase en los vecinos (9–10 columnas numéricas de frecuencia).
- `clc_uniform_surroundings` (booleana).
Las columnas por dirección (N, S, E, W…) se pueden colapsar porque la dirección no aporta información relevante (cargas casi idénticas en PC1–PC5).

### 3.4 Conjunto D — Espacial puro (para hotspots)
`latitude`, `longitude` (convertidas a km con proyección, p. ej. EPSG:25830) y opcionalmente fecha como día continuo.

### 3.5 Variables solo para perfilado (no entran al algoritmo)
`frp`, `confidence_cat`, `confidence_num`, `ccaa`, presupuestos y ratios por ha, `year`, `riesgo_antropico_alto`, `nivel_relativo`, `indice_meteorologico`.

---

## 4. Análisis propuestos

### 4.1 Tipología de condiciones de incendio (clustering principal)
- **Datos:** conjuntos A, B y C (tres corridas, para ver qué aporta cada bloque).
- **Algoritmos:**
  1. **K-means / MiniBatchKMeans** sobre PCs (línea base rápida), k = 2–12.
  2. **Gaussian Mixture (GMM)** con covarianza `full` y `diag`; selección de k por BIC. Da probabilidades de pertenencia, útil porque los regímenes se solapan.
  3. **Clustering jerárquico (Ward)** sobre una muestra de 5.000 filas para ver la estructura del dendrograma y estimar k.
  4. **K-prototypes** (numéricas + categóricas) o **Gower + HDBSCAN** para el conjunto C con `clc_class` nominal.
- **Selección de k:** silhouette, Calinski-Harabasz, Davies-Bouldin, BIC (GMM), codo de inercia. Se espera que silhouette sea baja (datos continuos, sin fronteras nítidas); decidir k por estabilidad e interpretabilidad, no solo por la métrica.
- **Interpretación:** tabla de medias por cluster (en unidades originales), mapa de latitud/longitud por cluster, distribución mensual por cluster, mix de CCAA y de `clc_class`.
- **Validación externa (a posteriori):** ¿difieren los clusters en `frp` (Kruskal-Wallis + tamaño de efecto, no solo p-valor)? ¿En `confidence_cat`? ¿En `ccaa`? Cuánta información mutua comparten con `ccaa` y con `month` (si es muy alta, el clustering solo redescubre región/estación).

### 4.2 Detección de anomalías (condiciones extremas)
- **Isolation Forest**, **Local Outlier Factor** y **distancia de Mahalanobis robusta** (MinCovDet) sobre el conjunto A/B.
- Tasa de contaminación: no fijarla a ojo; revisar la distribución de scores y trabajar con el percentil 1–2 %.
- Contraste: ¿los anómalos coinciden con el top-15 % de FRP del mapa 5.3, o con CCAA/meses concretos (olas de calor, viento extremo)? Cruzar con los atípicos IQR de 01b 2.5 para ver cuáles son físicos (VPD alto con viento) y cuáles artefactos (sensor, imputación).

### 4.3 Reducción de dimensionalidad y visualización
- **PCA** (ya hecho para meteorología) → mantener 7 PCs como representación estándar.
- **UMAP** (n_neighbors 15–50, min_dist 0,1) y **t-SNE** sobre A/B/C, coloreado por cluster, `frp`, mes, CCAA. Solo para explorar; no usar distancias del embedding como entrada de otros modelos sin validar.
- **MCA** (análisis de correspondencias múltiples) o **FAMD** para sustituir al one-hot+PCA de CLC; comparar varianza explicada y legibilidad con el resultado de 29 PCs.
- **Autoencoder** pequeño (opcional, solo si UMAP/PCA no bastan): 13 → 6 → 3 → 6 → 13, error de reconstrucción también como puntuación de anomalía.

### 4.4 Hotspots espacio-temporales
- **HDBSCAN** (o DBSCAN con eps por k-distancia) sobre (x_km, y_km, día) con escala temporal elegida (p. ej. 1 día ≈ 5–10 km) para agrupar detecciones del mismo evento.
- Resultado: número de clusters de evento, tamaño, duración, FRP máximo. Sirve para (a) estimar cuántos "incendios" reales hay detrás de 30 k detecciones y (b) construir una tabla a nivel de evento con meteorología promedio.
- Comparar con clusters de §4.1: ¿un evento recorre varios regímenes meteorológicos?

### 4.5 Variante con virulencia en la entrada
Repetir §4.1 incluyendo `log1p(frp)` como entrada. Pregunta: ¿aparecen regímenes "baja potencia / alta potencia" dentro de las mismas condiciones? Comparar con la versión sin FRP mediante ARI entre ambas particiones.

### 4.6 Reglas de asociación (opcional, exploratorio)
Discretizar (terciles) VPD, viento, humedad del suelo, `franja_horaria`, `clc_class` y buscar combinaciones frecuentes con FRP alto (Apriori/FP-Growth). Valor interpretativo, bajo valor predictivo.

---

## 5. Validación sin etiquetas

| Control | Cómo | Criterio de aceptación |
|---|---|---|
| Estabilidad | 30 remuestreos del 80 % + ARI entre particiones | ARI medio > 0,7 para el k elegido |
| Estabilidad espacial | Excluir una CCAA (o bloque de latitud) cada vez, reasignar y comparar | Los clusters no desaparecen al quitar una región dominante |
| Robustez al preprocesado | Repetir con StandardScaler vs RobustScaler, con/sin `boundary_layer_height`, con/sin log | ARI > 0,6 entre variantes; si no, la solución depende del preprocesado |
| Independencia de artefactos | Información mutua con `ccaa`, `month`, `year`, `daynight` | Reportarla; si MI normalizada > 0,5 con una sola variable, el cluster es esa variable |
| Sentido físico | Medias por cluster coherentes (p. ej. VPD alto ↔ humedad baja ↔ ET0 alta) | Revisión manual del equipo |
| Utilidad externa | Diferencias en FRP y `confidence_cat` entre clusters (tamaño de efecto) | Efecto pequeño es aceptable; se reporta tal cual |
| Réplica en datos separados | Ajustar con 2023, asignar 2024 (y viceversa) | Perfiles de cluster similares entre años |

Nota: como los datos tienen autocorrelación espacio-temporal, las divisiones aleatorias de filas inflan la estabilidad; preferir las divisiones por CCAA o por año.

---

## 6. Orden de trabajo

1. **Preparación (cuaderno nuevo `03a_preparacion_no_supervisado`)**
   Corregir `ratio_prevencion_extincion`; definir listas `ENTRADA_*` y `PERFILADO`; transformaciones log; decidir imputación de capa límite; construir features de entorno CLC (§3.3); guardar `data/processed/unsupervised/features_*.csv`.
2. **Clustering meteorológico (`03b_clustering`)**
   §4.1 con el conjunto A, luego B y C; tablas de perfil; mapas.
3. **Anomalías y embeddings (`03c_anomalias_embeddings`)**
   §4.2 y §4.3.
4. **Hotspots (`03d_eventos_espaciotemporales`)**
   §4.4 y tabla de eventos.
5. **Validación y síntesis (`03e_validacion`)**
   §5, comparación entre variantes, elección de la tipología final.
6. **Reintegración con el cuaderno 02**
   Si la tipología es estable, añadir `cluster_meteo` y `score_anomalia` como columnas candidatas del futuro modelo de ocurrencia (sin etiquetas de test en el ajuste).

---

## 7. Entregables

- `cluster_labels.csv` (id de fila, k elegido, probabilidades GMM, score de anomalía, id de evento).
- Tabla de perfiles por cluster (medias, medianas, % por CCAA, % por mes, FRP).
- Figuras: scree/BIC, silhouette, UMAP coloreado, mapa por cluster, estacionalidad por cluster, anomalías sobre el mapa.
- Nota metodológica con las limitaciones de §2 (solo detecciones, dos años, sesgo regional).

---

## 8. Límites de interpretación

- Los clusters describen **cómo son las condiciones cuando hay detección**, no cuándo es probable que haya incendio. Para eso hacen falta días de fondo sin fuego (la vía de `occurrence.py` del proyecto).
- Detecciones satelitales ≠ incendios forestales: incluyen quemas agrícolas y prescritas; el filtro `type == 0` y `clc_class != 1` ayuda, pero no las elimina.
- Los valores meteorológicos son de reanálisis/Open-Meteo en un punto, no mediciones locales.
- Un cluster "bonito" en UMAP no es evidencia por sí solo; la prueba es la estabilidad (§5).
