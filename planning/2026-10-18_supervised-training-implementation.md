# Supervised training — implementation plan, revision (Plans C–D / notebook 04)

**Date**: 2026-10-18. **Status**: working plan for the modeling side of block 04
(`notebooks/04_aprendizaje_supervisado.ipynb`). Refines Steps 3–6 of
`planning/2026-10-17_supervised-training-implementation.md` (whose Steps 1–2 —
k12 persistence and the reintegration module — are done ✅) with the decisions
taken 2026-10-18. Strategy lives in `planning/plan_supervised_implementation.md`;
placement rules in `planning/background_data_integration.md`. Decisions closed
here are marked **[DECIDED]**; open ones **[OPEN]**.

**Objective**: unchanged — a *risk ranking* of fire occurrence from everyday
(non-satellite) conditions. Positives = FIRMS detections
(`data/processed/expanded/firms_spain_final_lags_index.csv`, raw units, key
`detection_id`); negatives = background CCAA×day rows. Primary metric **PR-AUC**,
secondary ROC-AUC; no calibrated probabilities (see
`plan_supervised_implementation.md` §Plan D).

## Scope & boundary

- **Team (enrichment side)**: builds the enriched background file and all new
  (derived) features — weather → CLC → `ccaa` → budgets → derived columns;
  assigns `background_id`; applies the leakage drop; reports NaNs. Re-runs the
  positives through the same shared feature functions so parity is by
  construction (shared functions, separate frames — background is never merged
  into `firms_spain_final_lags_index.csv`).
- **This document (modeling side)**: consumes the enriched file; owns the
  loader/validator, the modeling frame, the model + ablation units, notebook 04,
  its tests, and the config/docs updates.

## 1. Input contract — enriched background file

Target: `data/processed/expanded/background_days.csv`, key `background_id`.

The file must provide, for every row, the columns `occurrence_features` /
`assemble_modeling_frame` read (raw inputs; `occurrence_features` derives
`wd_sin`/`wd_cos` itself):

location (`latitude`, `longitude`), calendar (`month`, `day_of_year`,
`month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos`), `acq_date`
(for `add_season`), anthropogenic (`riesgo_antropico_alto`), the 15 weather
variables incl. raw `wind_direction_10m`, CLC (`clc_class` + 8 neighbours +
`clc_uniform_surroundings`), `ccaa`, budgets (`prevention_current`,
`prevention_prev{N}`, `extinction_prev{N}`), and the derived indices
(`ratio_prevencion_extincion`, `euros_prevencion_por_ha`,
`euros_extincion_por_ha`, `potencial_propagacion`, `indice_riesgo_meteo`,
`indice_riesgo_meteo_log`, `estres_termico_vegetacion`).

Hard requirements:

- No satellite columns; `type` dropped or 0.
- Every row ≥ ~1 km from a same-day detection; the drop count is reported.
- `background_id` unique; `season` derivable from `acq_date`.
- Per-column NaN report.
- BLH imputed **per file with that file's own median** [DECIDED 2026-10-18].

## 2. Decisions log (2026-10-18)

- [DECIDED] Negatives ratio `background_ratio: 2`; if the enriched pool is
  smaller, use **all** available rows.
- [DECIDED] BLH median applied per file (positives with the positives' median,
  background with the background's median).
- [DECIDED] Leakage: drop background rows within ~1 km of a same-day detection;
  the modeling side **re-verifies** the rule (does not trust the build).
- [DECIDED] Reintegration candidates (`cluster_meteo`, `score_anomalia`) are
  produced in notebook 04 via `processing/reintegration.py`
  (`fit_background_scorer`), not in the enriched file.
- [DECIDED] Background file path/key: `data/processed/expanded/background_days.csv`,
  `background_id`.
- [DECIDED] Categorical encoding (`ccaa`, `clc_class`, candidate `cluster_meteo`)
  is one-hot, **fit on the train fold**, aligned across folds, unseen test
  categories → all-zero.
- [DECIDED] Ablation promotion uses a **margin + bootstrap CI** over folds, not
  a bare sign test; per-fold min reported.
- [DECIDED] PCA provenance → TODO; reuse the current PCA for now.
- [OPEN] `cluster_meteo` as one-hot categorical (default) vs 12 posteriors.
- [OPEN] Candidate model set + tuning budget (default: the provisional three, no
  tuning).
- [OPEN] Deployment unit / calibration (Plan D) — ranking-only default.

## 3. Workstreams

### WS1 — Consumption layer

`src/wildfire/processing/occurrence.py` (+ `tests/test_occurrence.py`):

- `load_background_days(path=None)` — config `output.background_days`; asserts
  `background_id`.
- `validate_background_days(background, positives, leakage_max_km=1.0)` —
  fail-closed: required columns present, no satellite columns, unique key,
  per-column NaN report, and a leakage re-check (spatial index over ~1 km /
  same-day collisions; raises on a hit unless overridden).
- `sample_background(background, ratio, seed)` — seeded; sorted by
  `background_id` for reproducibility; `n = ratio × n_positives`, or all rows if
  fewer are available.
- Reuse `assemble_modeling_frame` + `add_season` on the combined frame.

### WS2 — Model units

New package `src/wildfire/analysis/` (`__init__.py`, `occurrence_model.py`;
`tests/test_occurrence_model.py`):

- `model_candidates(seed=42)` — `LogisticRegression` (interpretable floor),
  `RandomForestClassifier`, `HistGradientBoostingClassifier`; sklearn only,
  sensible defaults, fresh instance per name per fold.
- `fold_scaler_fit` / `fold_scaler_apply` — standardize continuous columns only.
- `categorical_to_dummies(frame, cols, train_columns=None)` — fit on the train
  fold; aligned across folds; unseen test categories → all-zero. Default
  `cols = [ccaa, clc_class]` (+ `cluster_meteo` when candidates are used).
- Column-policy constants derived from `EVERYDAY_FEATURES` (continuous vs
  categorical).

### WS3 — Ablation (promotion gate)

New `src/wildfire/analysis/ablation.py` (+ `tests/test_ablation.py`):

- `ablation_verdict(with_metrics, without_metrics, margin=..., seed=42)` —
  per-fold ΔPR-AUC, mean, **bootstrap CI** (seed 42), per-protocol min, and a
  `promoted` decision (mean Δ > margin and CI excluding ~0, not negative in any
  protocol); negative controls (`month` / `ccaa` one-hot) run through the same
  function.

### WS4 — Notebook 04

Replace the stale two-cell skeleton; bilingual; anchored IO (`save_csv`/
`load_csv`); config paths only; one cell per step:

0. Title/objectives + rule reminder (blocked splits, seed 42, raw units, no
   satellite columns).
1. Load positives + background; `validate_background_days`; `sample_background`;
   `assemble_modeling_frame`; `add_season`; sanity cells (label balance 1:2,
   per-CCAA/per-season counts of both classes, NaN report shown).
2. Baseline fits + blocked-split evaluation (`season`, `ccaa`) → fold-metrics
   table **with baseline PR-AUC** and per-fold mean/min.
3. Candidate model comparison table.
4. Reintegration: `fit_background_scorer(positives)` → `cluster_meteo` +
   `score_anomalia` for both classes; rebuild the frame with
   `pos_candidates`/`neg_candidates`.
5. Ablation + verdict (`promoted` / `not promoted`).
6. Outputs: `data/processed/predictions/occurrence_scores.csv`, fold-metrics
   summary CSV, PR curves, per-CCAA PR-AUC choropleth, permutation importance for
   the floor model, package-versions cell, methods-note paragraph (ranking not
   probability, sampling assumptions, quota-bounded weather, PCA provenance TODO,
   empirical CCAA-day fire rates).

### WS5 — Reintegration consumption

Reuse `processing/reintegration.py` / `K12Typology`. Enforce a **single**
`score_anomalia` definition applied to both classes through
`BackgroundScorer.score` (never mixing the `anomalias_a.csv` path for one class
and the refit path for the other). PCA provenance stays [OPEN] (TODO).

### WS6 — Docs / config / TODO

- `configs/project.yaml`: add `output.background_days`,
  `modeling.occurrence.leakage_max_km`.
- `AGENTS.md`: `src/wildfire/analysis/` modules, notebook-04 outline, background
  chain entry.
- TODO: PCA provenance (verify `fit_weather_pca` reproduces the
  `features_a.csv` / 03b space).

## 4. Config & docs changes

As WS6; no hardcoded paths/params in source (all via `configs/project.yaml`).

## 5. Sequencing

WS1 + WS2 + WS3 are pure code with no data dependency → can start immediately.
WS4 needs WS1–WS3 and the enriched file, and runs on a partial pool (metrics
improve as the fill progresses). WS6 last.

## 6. Verification

- Full `pytest tests/` green (494 collected + the new tests).
- `ruff check src/ tests/` and `mypy src/` at their baselines (28 / 14
  pre-existing) — no new errors.
- Notebook 04 executes headless end-to-end on the enriched file.

## 7. Residual risks

- **PCA provenance [OPEN]**: if today's PCA differs from the `features_a.csv`
  provenance, `cluster_meteo` may be scored in a different space.
- **Unfrozen pool**: the exact modeling frame changes while the background fill
  runs; results are comparable only once the pool is frozen — report the PR-AUC
  baseline (= prevalence) and per-fold min on every run.
- **Small / single-class folds**: low-positive CCAAs and exterior quarters can
  produce noisy or single-class folds; guards in WS1/WS4 + per-fold reporting
  (`rank_metrics` requires both labels).
- **Encoding**: one-hot must be fold-fit (WS2) or it leaks across folds.

## 8. Open items carried

- `cluster_meteo` categorical (12 levels, default) vs 12 posteriors.
- Candidate model set + tuning budget (optuna [OPEN], no new deps by default).
- Deployment unit & calibration route (Plan D — ranking-only default; CCAA-day
  probability as the actionable upgrade).
