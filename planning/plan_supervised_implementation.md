# Supervised section plan — 04 occurrence model

**Status**: working plan for `notebooks/04_aprendizaje_supervisado.ipynb` (currently a
two-cell skeleton that predates the occurrence contract — its outline is stale and
will be replaced). Satisfies `plan_unsupervised_implementation.md` §7B/§8.6 (ablation
contract) and §3 of `plan_unsupervised_next_implementations.md` (reintegration).
Decisions closed here are marked **[DECIDED]**; open ones, **[OPEN]**.

**Objective**: a *risk ranking* of fire occurrence — "when/where is a fire likely" —
from everyday (non-satellite) conditions. Positives = FIRMS detections
(`data/processed/expanded/firms_spain_final_lags_index.csv`, 23,142 rows, raw units,
key `detection_id`). Negatives = random background CCAA×day rows (**do not exist
yet** — the critical-path blocker, §1). Primary metric **PR-AUC** (`rank_metrics`),
secondary ROC-AUC. No calibrated probabilities.

## 0. Hard rules (from the plans, non-negotiable)

1. **Splits**: `blocked_splits` only — LeaveOneGroupOut over `season` (YYYY-Qn) and
   over `ccaa`. No pure-random split as the only test (autocorrelation inflation).
2. **Scalers fit inside each experiment/fold** (hard rule 4.3). Raw units on disk;
   `firms_spain_final_normalized.csv` is descriptive-only and never an input.
3. **Seed 42 everywhere** (`CLUSTERING_SEED`, `occurrence_settings.random_state`).
4. **sklearn + scipy only** — xgboost/lightgbm/catboost/imblearn/shap/optuna are not
   installed; any new dep is **[OPEN]** per working-plan §6 and needs approval.
5. Contract discipline: features come from `occurrence_features`
   (`EVERYDAY_FEATURES` allowlist, fixed order); satellite columns in →
   `assemble_modeling_frame` raises; package versions recorded with every output.
6. Reintegration candidates (`cluster_meteo`, `score_anomalia`) are **candidates**:
   they enter the ablation, and are promoted only if ΔPR-AUC > 0 under blocked
   splits. Until then "exist but unproven" in the methods note.

## 1. Background days — the critical path [OPEN: quota]

Contract: `background_skeleton(ccaa, start, end, n_per_day, seed)` → CCAA×date grid,
weather/CLC filled afterwards, `assemble_modeling_frame(positives, background)`.
`background_ratio: 2` → **~46k background rows** for 23,142 positives.

- **Locations**: `background_skeleton` emits only `(ccaa, acq_date)`. Sample `n_per_day`
  point locations **inside each CCAA polygon** (geopandas `sample_points`, seed 42),
  store lat/lon, then run the **batched weather machinery**
  (`enrichment/weather_batch.py`, 500-row POST batches). ~46k rows ≈ 92 batches ≈
  12 h wall-clock at the 450 s/batch pacing — **spends real Open-Meteo quota**:
  team go/no-go required **[OPEN]**.
- **Quota mitigation options [OPEN]**: (a) `n_per_day=1` with ratio 2 achieved by
  *reusing* each background row for up to 2 positives (assemble allows 1:2); (b)
  background only where/when it discriminates (fire-season months × inland CCAA);
  (c) reduce ratio via config (`background_ratio` is a config knob). Default plan:
  full 46k, option (a) as fallback.
- **Fill order** mirrors the pipeline: weather (batched) → CLC
  (`enrichment/clc_enrichment.py`) → `assign_ccaa` sanity check (should reproduce
  the skeleton's ccaa) → `add_ccaa_budget_sums`. Save
  `data/processed/expanded/background_days.csv` (key `background_id`), never mixing
  with raw detections.
- **Validation**: `validate_enriched` weather warnings; CCAA reproduction rate;
  BLH nulls imputed exactly as notebook 01 (median) — documented, since the same
  imputation must be reproducible for scoring.

## 2. Feature frame

1. `occurrence_features(positives)` and `occurrence_features(background)` separately
   (identical 45-col schema, fixed order), then
   `assemble_modeling_frame` (labels 1/0, satellite check).
2. Add `season` via `add_season` for the season-blocked CV.
3. Sanity cell: label balance (1:2), per-CCAA and per-season counts of both classes,
   no-NaN policy per column (BLH median-imputed flag kept as a column? **[OPEN]** —
   default: no, impute silently as 01 does, note in methods).

## 3. Reintegration candidates (from 03b/03c/03e)

**[DECIDED] k12 is the elected typology** (03e verdict; k5/k7 rejected). Gaps to close
first — all three are preconditions for the ablation:

1. **Persist k12 labels**: `cluster_labels.csv` on disk only has k5. 03e refits
   GMM k12 inline; extract those labels (+ posteriors) into
   `data/processed/unsupervised/cluster_labels_k12.csv` keyed by `detection_id`
   (small helper cell or script; records the refit policy: full-covariance GMM k=12
   on StandardScaler(PC1–7), seed 42).
2. **Score background rows with the same fitted objects** (contract §3.2): refit
   scaler+GMM k12 on *positives only*, then `.predict` background PCs →
   `cluster_meteo` for both classes. Policy **[DECIDED]**: refit on positives;
   background scored by the fitted object, never included in the fit.
3. **`score_anomalia` definition [OPEN]**: no continuous consensus score exists on
   disk (only boolean `consenso_flag`). Default: row-wise mean of the three z-scored
   `*_score` columns from `anomalias_{a,b}.csv`, min-max rescaled to [0,1]; for
   background rows, refit IForest/LOF/Mahalanobis on positives and score background
   (same fitted-object policy). Document the choice in the methods note.
4. **Contract extension**: add `cluster_meteo`, `score_anomalia` to a
   *candidate-features* path (separate from `EVERYDAY_FEATURES` until promoted) +
   tests in `tests/test_occurrence.py` (candidate columns optional, order stable).

## 4. Model & evaluation

- **Candidates [DECIDED provisional]**: `LogisticRegression` (interpretable floor),
  `RandomForestClassifier`, `HistGradientBoostingClassifier` — sklearn only, seed 42.
  No tuning budget initially: sensible defaults, document; optuna **[OPEN]**.
- **Protocol**: for each of `season` and `ccaa` blocked splits → fit on train folds
  (scaler fit inside fold), score test, collect `rank_metrics`. Report per-fold
  PR-AUC (mean/min), confusion-free ranking curve data, and per-CCAA/per-season
  performance tables.
- **Figures**: PR curves per split group, per-CCAA PR-AUC choropleth, feature
  importance (permutation, on one season split) for the floor model.
- **Outputs**: `data/processed/predictions/occurrence_scores.csv`
  (id, label, score, held-out group per protocol), summary CSV of fold metrics;
  package versions cell (`sklearn.__version__` etc.) per hard rule 4.

## 5. Ablation contract (§7B/§8.6) — the gate for promotion

For the best base model, under **both** blocked-split protocols, with vs without
`cluster_meteo` + `score_anomalia`:

- ΔPR-AUC per fold; promotion requires **mean ΔPR-AUC > 0** (and not negative in any
  protocol). Negative controls: same ablation for `month` and `ccaa` one-hot columns
  (sanity that the candidates beat trivial additions).
- Verdict cell: **promoted / not promoted**, written into the methods note either way.

## 6. Notebook structure (04, one cell per step)

0. Title/objectives (bilingual), rule reminder (splits, seeds, raw units).
1. Load positives + background, assemble frame, sanity cells.
2. Baseline fits + blocked-split evaluation (season, ccaa).
3. Candidate model comparison table.
4. Reintegration: k12 labels, background scoring, candidate columns.
5. Ablation + verdict.
6. Predictions + figures + methods-note paragraph (honest limits: ranking not
   probability, background sampling assumptions, quota-bounded weather).

## 7. Tests to add

- Background scoring: fitted-on-positives objects score background (mock tiny frames).
- Candidate-columns path in `occurrence_features` (order, missing-column behavior).
- Ablation harness: ΔPR-AUC sign flips as expected on a synthetic separable column.
- Keep `pytest tests/test_occurrence.py` green (23 tests currently).

## 8. Open decisions checklist

- [ ] **Team go/no-go on Open-Meteo quota** for ~46k background rows (§1) — blocks
      everything; fallback (a) reduces to ~23k rows.
- [ ] Background `n_per_day` / spatial sampling inside polygons (seeded).
- [ ] `score_anomalia` exact formula (default in §3.3).
- [ ] Candidate model set + any tuning budget.
- [ ] Whether `cluster_meteo` enters as categorical (12 levels) or posteriors
      (12 extra numerics) **[OPEN]** — default: categorical + one-hot inside fold.
- [ ] New dependencies (default: none).

## 9. Risks / gotchas

- **Quota/time**: background weather is the long pole; resume CSV exists
  (`weather_batch.py`), plan an interruptible run + `logs/`.
- **Positional joins**: 03a–03e CSVs share no key except the newly-noted
  `detection_id` in the lags CSV — 04 must join **on `detection_id`**, never by row
  order (the audit fragility ends here for the supervised side).
- **Background leakage traps**: background days must exclude real detections' exact
  (lat, lon, day) cells; enforce a minimum distance / day-exclusion rule and count
  collisions **[OPEN]**, default: drop background rows within ~1 km of a detection on
  the same day.
- **CCAA imbalance**: some CCAAs have few positives; ccaa-blocked folds will be
  noisy — report per-fold, don't aggregate away.
