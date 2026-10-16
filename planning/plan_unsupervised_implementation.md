# First unsupervised-learning implementation — Spain wildfires (FIRMS 2023–2024)

**Status**: execution plan, and the working version going forward. Consolidates
`plan_aprendizaje_no_supervisado_claude.md` (figures verified by execution against the repo) and the salvageable
parts of `plan_no_superv_deepseek.md`. Decisions closed during review are marked **[DECIDED]**; open ones,
**[OPEN]**.

**Working base**: `data/processed/expanded/firms_spain_final_lags_index.csv` (30,412 rows × 60 columns).
Not `firms_spain_final_normalized.csv`: its scaler saw the whole dataset and is descriptive only.

---

## 1. What this is (for everyone)

When the satellite detects a fire, that day has conditions: heat, wind, dryness, vegetation type, time of
year. This block groups the 30,412 detections by **similarity of conditions** to answer three questions:

1. Are there distinguishable *types* of fire days? (e.g. "heatwave with wind" vs "damp winter burn"). That is
   the **conditions typology** (clustering). Note: these groups describe what days *with* fire look like;
   whether that description helps prediction is decided by §7B, not by the clustering.
2. Which detections are *rare* relative to the rest? (candidates for extreme events or sensor artefacts).
   That is **anomaly detection**.
3. How many **real fires** stand behind 30,412 detections? (one fire is detected over several days). Those
   are the **spatio-temporal events**.

## 2. What this block will NOT deliver (honest limits)

- **No fire probability.** We only have days *with* detections; background days without fire are missing
  (via `src/wildfire/processing/occurrence.py`, pending). The groups describe *what fire days look like*,
  not *when fire will occur*.
- **No separation of wildfires from agricultural burns.** The `type == 0` + `clc_class != 1` filter helps
  but does not eliminate them; Asturias/Cantabria in winter weigh heavily, so read the groups with that bias.
- **Two years are not climate.** Everything is descriptive of 2023–2024; do not generalise to other seasons.
- **A "pretty" group in a plot is not evidence.** Stability proves the structure exists (§7A), not that it is
  useful for prediction (§7B).
- **Stable does not mean useful.** A by-month partition is perfectly stable (deterministic) and useless for
  anything: groups repeating does not prove they help predict fires.

## 3. Starting point (facts verified by execution)

| Fact | Value | Consequence |
|---|---|---|
| Size 30,412 × 60 | fits in memory; distance matrices fit without sampling | GMM/HDBSCAN usable with no shortcuts |
| Weather PCA | 14 vars → 7 PCs at 90.9 %; PC1 = dryness/heat | set A = 7 PCs (decorrelates the 5 collinear PC1 vars) |
| `corr(PC1, indice_meteorologico)` = 0.916 | near-redundant | use one or the other, never both |
| Kruskal CLC vs FRP | H = 1503.7; p ≈ 0 | land cover carries signal: justifies set C |
| `confidence_cat` | n 85 % / h 10 % / l 5 % | external validation, never an input |
| `rain` ≈ `precipitation` | means 0.017968 vs 0.017981 | `rain` out, always |
| BLH | 4,996 NaN (16 %) imputed with the **global median** **[DECIDED]** | kept as is; the imputation spike is documented, not re-imputed |
| `ratio_prevencion_extincion` | **bug fixed** (was `prev/(prev+ε)` ≈ 1.0; now `prev/(ext+ε)` ∈ [0.402–0.942]) | column valid since the 01b regeneration |
| Budgets | 34 unique values (= CCAA × year) | profiling only, never inputs (else groups = regions) |
| Regional split | Asturias 5,206 … La Rioja 102 | watch bias; leave-one-CCAA test (§7) |
| 02c MLP | predicts `confidence_cat` (a sensor artefact), parked | not a reference for anything; do not compare against it |

## 4. Hard rules (non-negotiable in this phase)

1. **No satellite data in the inputs.** `brightness`, `brightness_ir`, `scan`, `frp`, `confidence_num` stay
   out (`SATELLITE_BLOCKLIST` from `occurrence.py`). `frp` and `confidence_cat`/`confidence_num` are for
   profiling and post-hoc validation only.
2. **One representation per concept.** No `rain`; no raw `month`/`day_of_year` (yes their sin/cos); no
   `is_day`/`daynight` (yes `franja_horaria`); no raw direction (yes `wd_sin`/`wd_cos`); no composite indices
   mixed with their base variables (documented exception: sets B/C use indices + meteorology because the goal
   there is interpretability, and they are compared against A).
3. **Every scaler is fit inside each experiment**; never reuse the `normalized.csv` scaler. StandardScaler vs
   RobustScaler comparison as the first test (§7).
4. **Fixed seeds** (`random_state=42`, the project one) and package versions recorded with every output.

## 5. Feature sets (closed lists)

Exact names from `lags_index.csv`. `componentes_disponibles` (=12, constant), `nivel_relativo`,
`indice_meteorologico` and `*_lag_*`/`*_prev_*` lags **enter no set** (constants, redundant with PC1, or
structural NaN).

### Set A — meteorology (14 numerics or their 7 PCs; recommended: PCs)

The 12 from `WEATHER_PCA_BASE` (`pca_weather.py:43`) + circular pair:

```
temperature_2m, relative_humidity_2m, vapour_pressure_deficit, et0_fao_evapotranspiration,
shortwave_radiation, boundary_layer_height, surface_pressure, wind_speed_10m, wind_gusts_10m,
precipitation, soil_moisture_0_to_7cm, soil_moisture_7_to_28cm, wd_sin, wd_cos
```

(`wd_sin`/`wd_cos` derive from `wind_direction_10m` via `AddWindCircular`, as in the PCA; the raw column
stays out.) Transforms: `log1p` on `precipitation`, `wind_gusts_10m`, `vapour_pressure_deficit`.

### Set B — A + seasonality and time of day

A + `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos` (unscaled: already in [-1, 1]) +
`franja_horaria` (encoded; the only representative of time of day). Goal: separate "summer afternoon" from
"winter early morning".

### Set C — B + land use (to build in 03a)

B + collapsed CLC neighbourhood (the 8 directions add nothing: near-identical loadings in the 02c PCA):

- `clc_class` (centre pixel, nominal — only for methods that accept it).
- `pct_vecinos_igual_centro` (0–1), `n_clases_distintas_vecinos` (1–8), `clc_uniform_surroundings` (exists).
- Frequency of each class among the 8 neighbours (`freq_vecina_<clase>` columns, observed classes only).

### Set D — pure spatial (hotspots)

`latitude`, `longitude` converted to km (EPSG:25830) + date as continuous day. Only for §6.3.

### Profiling / validation only (enter no algorithm)

```
frp, brightness, brightness_ir, scan, confidence_num, confidence_cat, ccaa, year, month,
day_of_year, daynight, is_day, acq_time, acq_date, type, rain, wind_direction_10m,
prevention_current, prevention_prev3, extinction_prev3, ratio_prevencion_extincion,
euros_prevencion_por_ha, euros_extincion_por_ha, potencial_propagacion, indice_riesgo_meteo,
indice_riesgo_meteo_log, estres_termico_vegetacion, indice_meteorologico,
riesgo_antropico_alto, nivel_relativo, componentes_disponibles
```

(`riesgo_antropico_alto` stays in profiling in phase 1; candidate for a C variant if the typology is stable.)

## 6. Analyses

All with **scikit-learn + scipy** (what is installed). **[OPEN]** adding `umap`/`hdbscan`/`kmodes` only if the
sklearn phase leaves questions unanswered; they are not required to start.

### 6.1 Conditions typology (main clustering)

Run on A, then B and C (three runs: see what each block adds), plus two **negative controls** (trivial
partitions: month-only with 12 groups, CCAA-only with 17 groups). The candidate must beat them (§7B); else it
rediscovered the calendar or the geography with extra steps.

1. **KMeans** k = 2–12 (fast baseline). Metrics: inertia (elbow), silhouette, Calinski-Harabasz,
   Davies-Bouldin. Silhouette is expected to be low (continuous data, no sharp borders): k is chosen by
   stability and interpretability, not by the metric.
2. **GMM** (`full` and `diag`), k by BIC. Gives soft membership (regimes overlap).
3. **Hierarchical Ward** on a 5,000-row sample (dendrogram → independent k estimate).
4. **KModes/KPrototypes or Gower+HDBSCAN** only if the dependencies are approved, for C with `clc_class`.

Interpretation per run: per-group means **in original units**, lat/lon map per group, monthly split, CCAA and
`clc_class` mix. External validation: do groups differ in `frp` (Kruskal + effect size, not just p-value) and
in `confidence_cat`? How much mutual information do they share with `ccaa` and `month`? (If normalised MI >
0.5 with a single variable, the group *is* that variable.)

### 6.2 Anomalies (extreme conditions)

**Isolation Forest**, **LOF** and robust **Mahalanobis** (`MinCovDet`) on A/B. Rate: 1–2 percentile of the
score distribution, never eyeballed `contamination`. Cross-check: do they match the top-15 % FRP (02b map
5.3)? Specific CCAA/months? Which are physical (high VPD + wind) and which are artefacts (BLH imputation)?

### 6.3 Spatio-temporal events

**DBSCAN/HDBSCAN** on (x_km, y_km, day), with 1 day ≈ 5–10 km. Output: event count, size, duration, max FRP +
an **event-level table** (averaged meteorology). Estimates how many real fires stand behind 30 k detections
and asks whether one event spans several regimes (§6.1).

### 6.4 Virulence-inside variant (dropped)

**[DROPPED 2026-10-16]**: repeating §6.1 with `log1p(frp)` as an input would
violate hard rule 4.1 (no satellite data in inputs). The severity-separation
checks (§7B: Kruskal on `frp` across groups + Cliff's δ) answer whether
high/low-power regimes separate without breaking the rule.

## 7. Validation (two blocks: stability ≠ utility)

Stability (§7A) answers whether the structure *exists*. Utility (§7B) answers whether it *helps prediction*.
A typology is accepted only if it passes **both** (7A ∧ 7B). Without background days, occurrence utility is
currently unmeasurable directly: §7B measures it indirectly (severity, non-triviality) and leaves an explicit
future ablation contract.

### 7A. Statistical stability (does the structure exist?)

| Check | How | Acceptance |
|---|---|---|
| Stability | 30 resamples at 80 % + ARI | mean ARI > 0.7 at the chosen k |
| Spatial stability | drop one CCAA at a time, reassign | no group disappears without a dominant region |
| Temporal replication | fit on 2023, assign 2024 (and vice versa) | similar group profiles across years |
| Preprocessing robustness | Standard vs Robust, with/without BLH, with/without log | ARI > 0.6 across variants |
| Physical sense | high VPD ↔ low humidity ↔ high ET0, etc. | manual review |

Note: no pure-random splits as the only test (spatio-temporal autocorrelation inflates them); CCAA/year
blocks rule.

### 7B. Predictive utility (would it help prediction?)

| Check | How | Acceptance |
|---|---|---|
| Severity separation | Kruskal across groups on `frp` + max pairwise Cliff's \|δ\| | at least one pair with \|δ\| ≥ `unsupervised.min_cliff_delta` (default 0.2, provisional: anchor against real baselines in 03a) |
| Negative controls | same separation for month-only and CCAA-only partitions | the candidate beats **both**; else it is rediscovered calendar/geography |
| Non-coincidence | candidate ARI vs month and vs CCAA | low on both; ARI ≈ 1 with the calendar = it *is* the calendar |
| Non-triviality | normalised MI with `ccaa`, `month`, `year`, `daynight` | reported; > 0.5 with a single variable = the group is that variable |
| Future ablation contract | `occurrence.py` + background days: ΔPR-AUC with/without `cluster_meteo` | deferred criterion; until then utility is indirect and declared as such |

## 8. Work order and deliverables

Following the repo pattern (index notebook + pieces):

1. **03a_preparacion_no_supervisado** — reusable `ENTRADA_*`/`PERFILADO` lists, `log1p`, `wd_*`, CLC
   neighbourhood (§5-C), documented BLH. Saves `data/processed/unsupervised/features_{a,b,c}.csv` +
   `perfilado.csv` with a common index. Includes the Standard vs Robust comparison (first ARI).
2. **03b_clustering** — §6.1 (A → B → C), profiles, maps.
3. **03c_anomalias_embeddings** — §6.2 (+ existing PCA as reference; embeddings only if deps approved).
4. **03d_eventos** — §6.3 + event table.
5. **03e_validacion** — §7A + §7B, cross-variant comparison, final typology.
6. **Reintegration** — if the typology passes 7A ∧ 7B: `cluster_meteo` and `score_anomalia` as candidate
   columns for the future occurrence model (fit without test labels); final promotion stays subject to the
   §7B ablation contract.

Deliverables: `cluster_labels.csv` (id, k, P(GMM), score, event id), profile table, figures (BIC/silhouette,
map per group, seasonality, anomalies), methods note with §2. The current 03 (placeholder "GONZALO") becomes
the markdown index of the pieces, as with 01 and 02.

## Appendix — what is adopted and dropped from the prior plans

- **From Claude**: full structure, sets A–D, validation, limits, 03a–e order. Applied corrections: counts
  (Asturias 5,206, Rioja 102), budgets = CCAA×year (34 values, not 17), MLP = parked artefact (not classic
  "leakage"), BLH at median **[DECIDED]**.
- **From DeepSeek**: KMeans/GMM/DBSCAN sketches, per-group profiling, anomaly characterisation. **Dropped**:
  `frp`/`brightness`/`confidence` as inputs (violates rule 4.1), the "57 columns" shape (stale), eyeballed
  `contamination`, SOM/UMAP as requirements, and the total absence of validation and of the clusters≠risk
  warning.
