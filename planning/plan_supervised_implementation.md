# Supervised section plan — 04 occurrence model

**Status**: working plan for `notebooks/04_aprendizaje_supervisado.ipynb` (was a
two-cell skeleton predating the occurrence contract; its outline is stale and will
be replaced). Satisfies `plan_unsupervised_implementation.md` §7B/§8.6 (ablation
contract) and §3 of `plan_unsupervised_next_implementations.md` (reintegration —
those plan files were archived 2026-10-17; this file is the live plan). Decisions
closed here are marked **[DECIDED]**; open ones, **[OPEN]**.

**Objective**: a *risk ranking* of fire occurrence — "when/where is a fire likely" —
from everyday (non-satellite) conditions. Positives = FIRMS detections
(`data/processed/expanded/firms_spain_final_lags_index.csv`, 23,142 rows, raw units,
key `detection_id`). Negatives = random background CCAA×day rows. **Dataset decided at 100k rows**
(`data/processed/synthetic/firms_spain_background_100k.csv`) — supersedes the
~46k estimate below. Primary metric **PR-AUC** (`rank_metrics`),
secondary ROC-AUC. No calibrated probabilities.

This document is divided into four development plans, in dependency order:

- **Plan A — Background days** (§1): the critical-path data acquisition. Blocks B–D.
- **Plan B — Feature frame & reintegration candidates** (§2–3): needs A's background
  rows and closes the k12/score gaps inherited from 03.
- **Plan C — Model, evaluation & ablation** (§4–7): the notebook proper, gated on B.
- **Plan D — Deployment unit & calibration** (§7b): a parallel, non-blocking track that
  shapes methods-note wording and future work.

## 0. Hard rules (from the plans, non-negotiable)

1. **Splits**: `blocked_splits` only — LeaveOneGroupOut over `season` (YYYY-Qn) and
   over `ccaa`. No pure-random split as the only test (autocorrelation inflation).
2. **Scalers fit inside each experiment/fold** (hard rule 4.3). Raw units on disk;
   `firms_spain_final_normalized.csv` is descriptive-only and never an input.
3. **Seed 42 everywhere** (`CLUSTERING_SEED`, `occurrence_settings.random_state`).
4. **sklearn + scipy only** — xgboost/lightgbm/catboost/imblearn/shap/optuna are not
   installed; any new dep is **[OPEN]** and needs approval.
5. Contract discipline: features come from `occurrence_features`
   (`EVERYDAY_FEATURES` allowlist, fixed order); satellite columns in →
   `assemble_modeling_frame` raises; package versions recorded with every output.
6. Reintegration candidates (`cluster_meteo`, `score_anomalia`) are **candidates**:
   they enter the ablation, and are promoted only if ΔPR-AUC > 0 under blocked
   splits. Until then "exist but unproven" in the methods note.

---

## Plan A — Background days: the critical path

**File names [DECIDED 2026-10-17]**:

- Weather resume CSV: `data/processed/synthetic/firms_spain_background_weather_partial.csv`
  (100k rows; 20k weather-complete at launch, refilling since 2026-10-17 via
  `scripts/enrich_weather_batch.py --csv-path …partial.csv --complete-path …final.csv`,
  logs in `logs/background_weather_batch.{log,err}`).
- Final weather CSV: `data/processed/synthetic/firms_spain_background_weather.csv`
  (published by `finalize_weather_csv` on completion; may be refused if BLH stays
  null upstream, as with the FIRMS file).
- **Loader contract: try the final CSV, fall back to the partial** — downstream
  (notebook 04, Plan B) must not block on the refill; BLH imputed with the median
  exactly as notebook 01.
- Post-weather outputs keep the names in §1 below: CLC/`ccaa`/budget enrichment
  saved as `data/processed/expanded/background_days.csv` (key `background_id`).

**Fill can run in parallel with the rest of the project**: the refill is a
background task; Plans B–D proceed on the partial with the loader fallback above.

Contract: `background_skeleton(ccaa, start, end, n_per_day, seed)` → CCAA×date grid,
weather/CLC filled afterwards, `assemble_modeling_frame(positives, background)`.

- **Locations**: `background_skeleton` emits only `(ccaa, acq_date)`. Sample `n_per_day`
  point locations **inside each CCAA polygon** (geopandas `sample_points`, seed 42),
  store lat/lon, then run the **batched weather machinery**
  (`enrichment/weather_batch.py`, 500-row POST batches). Remaining ~80k rows ≈
  160 batches — **quota go decision taken 2026-10-17**, refill running (see file
  names above).
- **Quota mitigation options [OPEN]** (moot while the 100k refill runs; relevant
  only if the run stalls on limits): (a) reuse each background row for up to 2
  positives (assemble allows 1:2); (b) background only where/when it
  discriminates (fire-season months × inland CCAA); (c) reduce ratio via config
  (`background_ratio` is a config knob). Default plan: full 100k, option (a) as
  fallback.
- **Fill order** mirrors the pipeline: weather (batched) → CLC
  (`enrichment/clc_enrichment.py`) → `assign_ccaa` sanity check (should reproduce
  the skeleton's ccaa) → `add_ccaa_budget_sums`. Save
  `data/processed/expanded/background_days.csv` (key `background_id`), never mixing
  with raw detections.
- **Leakage guard (moved here from §9 — enforced at build time, not at modeling)**:
  background days must exclude real detections' exact (lat, lon, day) cells; enforce
  a minimum distance / day-exclusion rule and count collisions **[OPEN]**, default:
  drop background rows within ~1 km of a detection on the same day.
- **Validation**: `validate_enriched` weather warnings; CCAA reproduction rate;
  BLH nulls imputed exactly as notebook 01 (median) — documented, since the same
  imputation must be reproducible for scoring.

**Done when**: `background_days.csv` exists, passes validation, and joins cleanly
with positives in `assemble_modeling_frame` at the intended 1:2 ratio.

---

## Plan B — Feature frame & reintegration candidates

### B.1 Feature frame

1. `occurrence_features(positives)` and `occurrence_features(background)` separately
   (identical 45-col schema, fixed order), then
   `assemble_modeling_frame` (labels 1/0, satellite check).
2. Add `season` via `add_season` for the season-blocked CV.
3. Sanity cell: label balance (1:2), per-CCAA and per-season counts of both classes,
   no-NaN policy per column (BLH median-imputed flag kept as a column? **[OPEN]** —
   default: no, impute silently as 01 does, note in methods).

### B.2 Reintegration candidates (from 03b/03c/03e)

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

**Done when**: modeling frame assembles with the two candidate columns available on
both classes, all joins on `detection_id`/`background_id` keys, new tests green.

---

## Plan C — Model, evaluation & ablation (notebook 04)

### C.1 Model & evaluation

- **Candidates [DECIDED provisional]**: `LogisticRegression` (interpretable floor),
  `RandomForestClassifier`, `HistGradientBoostingClassifier` — sklearn only, seed 42.
  No tuning budget initially: sensible defaults, documented; optuna **[OPEN]**.
- **Protocol**: for each of `season` and `ccaa` blocked splits → fit on train folds
  (scaler fit inside fold), score test, collect `rank_metrics`. Report per-fold
  PR-AUC (mean/min), ranking curve data, and per-CCAA/per-season performance tables.
- **Figures**: PR curves per split group, per-CCAA PR-AUC choropleth, feature
  importance (permutation, on one season split) for the floor model.
- **Outputs**: `data/processed/predictions/occurrence_scores.csv`
  (id, label, score, held-out group per protocol), summary CSV of fold metrics;
  package versions cell (`sklearn.__version__` etc.) per hard rule 4.

### C.2 Ablation contract (§7B/§8.6) — the gate for promotion

For the best base model, under **both** blocked-split protocols, with vs without
`cluster_meteo` + `score_anomalia`:

- ΔPR-AUC per fold; promotion requires **mean ΔPR-AUC > 0** (and not negative in any
  protocol). Negative controls: same ablation for `month` and `ccaa` one-hot columns
  (sanity that the candidates beat trivial additions).
- Verdict cell: **promoted / not promoted**, written into the methods note either way.

### C.3 Notebook structure (04, one cell per step)

0. Title/objectives (bilingual), rule reminder (splits, seeds, raw units).
1. Load positives + background, assemble frame, sanity cells.
2. Baseline fits + blocked-split evaluation (season, ccaa).
3. Candidate model comparison table.
4. Reintegration: k12 labels, background scoring, candidate columns.
5. Ablation + verdict.
6. Predictions + figures + methods-note paragraph (honest limits: ranking not
   probability, background sampling assumptions, quota-bounded weather).

### C.4 Tests to add

- Background scoring: fitted-on-positives objects score background (mock tiny frames).
- Candidate-columns path in `occurrence_features` (order, missing-column behavior).
- Ablation harness: ΔPR-AUC sign flips as expected on a synthetic separable column.
- Keep `pytest tests/test_occurrence.py` green (23 tests currently).

**Done when**: notebook 04 runs end-to-end, ablation verdict recorded, methods-note
paragraph written.

---

## Plan D — Deployment unit & calibration: "chances", not just ranking [OPEN]

Parallel track — does not block A–C, but the recommendation below feeds the
methods-note paragraph in C.3 step 6.

The model as specified outputs a **ranking score**: `f(weather, location, day) → risk`,
comparable across rows ("this day/place is riskier than that one") — usable for
alerts, prioritization and maps. It is **not** a calibrated probability, and cannot
become one by post-processing, because:

1. **Prevalence is artificial.** The 1:2 fire/background ratio is a sampling choice
   (`background_ratio: 2`), not the true base rate. Probabilities from a 33%-positive
   training set would massively overstate real risk; no calibrator fixes a base rate
   that was chosen, not observed.
2. The contract says so explicitly: *"el objetivo es un ranking de riesgo (PR-AUC),
   no una probabilidad calibrada: la prevalencia la fija el muestreo de negativos,
   no la naturaleza."*

**Routes to true "chances" [OPEN]** — decide the deployment unit early, because it
shapes background sampling (Plan A) and the methods note:

| Route | What it yields | Cost / caveat |
|---|---|---|
| Ranking only (default) | Comparable risk score, no probability claim | None — the plan as written |
| CCAA-day probability | P(≥1 fire in CCAA c on day d) — directly estimable from FIRMS counts against the full CCAA×day grid | Honest and actionable for planning; needs the empirical fire-day rate per CCAA, no extra quota |
| Point-level calibration | "P(fire at this point)" via Platt/isotonic + prior correction against an estimated true prevalence | Fragile: true point prevalence is near-zero and hard to estimate; odds are misleading at that granularity |
| Hazard/rate reframing | Expected fires per area per day (Poisson-style intensity) | Cleanest probabilistic semantics; a bigger modeling change than 04's scope |

**Recommendation [DECIDED for 04]**: build the ranking model as planned, but (a) keep
the scoring pipeline separable so a calibration stage can be added later, (b) report
empirical CCAA-day fire rates alongside model scores in the deliverables cell (cheap,
no extra quota), and (c) write the probability question into the methods note as an
explicit open follow-up rather than letting scores be misread as probabilities.

**Done when**: the deployment-unit decision is recorded (even if "ranking only") and
the methods note carries the fire-rate table and the open follow-up.

---

## Open decisions checklist

- [x] **Team go/no-go on Open-Meteo quota** for background rows (Plan A) —
      **decided 2026-10-17: go**; refill of `firms_spain_background_weather_partial.csv`
      (16k/100k at launch) running in background with auto-resume. Fallback (a)
      remains if limits stall the run.
- [ ] Background `n_per_day` / spatial sampling inside polygons (seeded).
- [ ] Background leakage rule: distance/day exclusion vs detections (Plan A).
- [ ] `score_anomalia` exact formula (default in B.2.3).
- [ ] Candidate model set + any tuning budget.
- [ ] Whether `cluster_meteo` enters as categorical (12 levels) or posteriors
      (12 extra numerics) **[OPEN]** — default: categorical + one-hot inside fold.
- [ ] New dependencies (default: none).
- [ ] Deployment unit & calibration route (Plan D) — ranking-only default; CCAA-day
      probability as the actionable upgrade; point-level "chances" not recommended.
- [ ] Leakage-exclusion distance (default ~1 km) and whether collisions are dropped
      or re-sampled.

## Risks / gotchas

- **Quota/time**: background weather is the long pole; resume CSV exists
  (`weather_batch.py`), plan an interruptible run + `logs/`.
- **Positional joins**: 03a–03e CSVs share no key except the newly-noted
  `detection_id` in the lags CSV — 04 must join **on `detection_id`**, never by row
  order (the audit fragility ends here for the supervised side).
- **Background leakage traps**: see Plan A — enforce and count collisions at build
  time, not after modeling.
- **CCAA imbalance**: some CCAAs have few positives; ccaa-blocked folds will be
  noisy — report per-fold, don't aggregate away.
