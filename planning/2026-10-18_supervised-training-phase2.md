# Supervised training — phase 2 implementation plan

**Date**: 2026-10-18. **Status**: WS1–WS5 done and green, WS6 partial, notebook 04
runs end-to-end (commit `8f6f10f`, status in
`planning/2026-10-18_supervised-training-status.md`). This file plans the **next**
part: hardening, closing the open decisions, and making the results publishable.
It continues the workstream numbering of
`planning/2026-10-18_supervised-training-implementation.md` (WS1–WS6) and must be
read with `planning/plan_supervised_implementation.md` (the strategy, hard rules
and Plan D).

**Objective (unchanged)**: a *risk ranking* of fire occurrence from everyday
(non-satellite) conditions. Positives = FIRMS detections
(`data/processed/expanded/firms_spain_final_lags_index.csv`, raw units, key
`detection_id`); negatives = enriched background CCAA×day rows
(`data/processed/expanded/background_days.csv`, raw units, key `background_id`).
Primary metric **PR-AUC**, secondary ROC-AUC; **no calibrated probabilities**.

## Current-state snapshot (2026-10-18 run)

- Background validated: **23,895 × 58**, key `background_id`, 0 duplicates,
  0 leakage, no NaN. `validate_background_days` passes.
- Sampling: requested `ratio 2` = 46,284 → **pool smaller → all 23,895 used**
  (effective ≈ 1.03:1). The pool is **not frozen**.
- Baseline PR-AUC (mean fold prevalence): **0.456**; best provisional model:
  `hist_gradient_boosting`.
- Reintegration candidates (`cluster_meteo` + `score_anomalia`): **not
  promoted** (ΔPR-AUC +0.0039, CI95% [−0.0033, +0.0116]); negative controls
  `month`/`ccaa` also not promoted (−0.0009 / +0.0000).
- `pytest tests/` = 550 passed, 2 deselected; touched files ruff/mypy clean.

**Consequence**: the ablation verdict is not yet a final answer — it was computed
on an unfrozen, undersized pool, and `cluster_meteo` provenance is unverified.
Phase 2 exists to remove those two caveats before any number is published.

## Scope & boundary

- **Team (data side)**: grows the enriched background pool toward
  `background_ratio: 2` (≥ 46,284 rows) and reports when the pool stops changing.
  No modeling code.
- **This document (modeling side)**: PCA-provenance verification, candidate
  encoding/definition, model set + tuning, pool-freeze reporting, the
  deployment-unit deliverable, and the docs/tests around all of it.

## Decisions to close in phase 2

- [ ] **Pool freeze rule**: publish numbers only once the pool is frozen at
      `ratio 2`, or the team declares it stopped (report actual ratio either way).
- [ ] **PCA provenance policy [OPEN]**: reuse the persisted 03b space vs refit —
      default: if `fit_weather_pca` does *not* reproduce `features_a.csv`,
      persist the pipeline and make `BackgroundScorer` reuse it.
- [ ] **`cluster_meteo` encoding**: hard labels one-hot (default) vs the 12 GMM
      posteriors — decide by the ablation gate, not by preference.
- [ ] **Model set + tuning budget**: default = provisional three + class weights,
      small hand grid per fold, seed 42, **no new deps**; optuna stays [OPEN].
- [ ] **Deployment unit**: ranking-only default; CCAA-day empirical rate as the
      cheap, actionable upgrade. Point-level calibration stays not recommended.
- [ ] **Ablation margin**: today `margin=0.0`; decide whether a small practical
      margin (e.g. +0.005) is required to promote.

## Workstreams

### WS7 — Freeze the pool & re-baseline

**Why**: every current metric is on an unfrozen ~1.03:1 pool; the PR-AUC baseline
(= prevalence) will move as the pool grows.

- Make notebook 04 report **pool adequacy explicitly** every run: requested
  `ratio × n_positives`, available rows, actual ratio, prevalence, and the
  per-protocol baseline PR-AUC with per-fold min. (The sampling print and the
  baseline table already exist — promote them into the methods/report cell and
  add an explicit "pool frozen: yes/no" line.)
- Coordinate the team fill to ≥ 46,284 rows (or a declared stop), then re-run and
  **write the new numbers into the status doc** (`…-status.md`) with the commit.
- Keep `background_ratio: 2`; the fallback-to-all behaviour in
  `sample_background` stays.

**Deliverable**: one re-baselined run with the pool state stated; status doc
updated. No source change beyond the reporting cell.

### WS8 — PCA provenance verification (unblocks `cluster_meteo`)

**Why**: `fit_background_scorer` (`src/wildfire/processing/reintegration.py:200`)
refits `fit_weather_pca` on the positives, while the persisted k12 typology was
fit on `features_a.csv` (03a: `fit_weather_pca` → PC1–7, saved as `features_a`;
`scripts/persist_k12_labels.py`). If the two spaces differ, positive
`cluster_meteo` labels in 04 do not match the persisted typology.

- Add a verification (test + small script): fit on `firms_spain_final_lags_index.csv`
  raw columns (`pca_input_columns()`) and compare PC1–7 to `features_a.csv`
  (correlation up to sign; PCA is sign-indeterminate); assert `n_components == 7`
  and cumulative variance ≈ 0.915 (`pca.target_variance: 0.90`).
- **Policy if it matches**: close the TODO with the test as evidence.
- **Policy if it does not**: persist the 03b pipeline (new `save_weather_pca` /
  `load_weather_pca`, path under `data/processed/unsupervised/`) and have
  `BackgroundScorer` **reuse** it instead of refitting; re-run 04.
- Close the PCA entry in TODO (`TODO.md`) and the methods-note [PENDIENTE] line in
  cell 14 of notebook 04.

**Deliverable**: provenance test green + TODO closed (or a recorded decision to
persist/reuse the pipeline).

### WS9 — Candidate reintegration, take 2

**Why**: the candidates were tested once, on an unfrozen pool, as hard-label
one-hot; that is not a fair final verdict. PLAN §B.2.3/§C.2 fixed the gate, not
the encoding.

- `cluster_meteo`: run the ablation with **one-hot hard labels** (current) vs
  **12 GMM posteriors** (`gmm_k12_full_p{j}` from `K12Typology`), same gate.
  Posteriors enter as 12 continuous columns (extend the candidate policy in
  `src/wildfire/analysis/occurrence_model.py`).
- `score_anomalia`: confirm the **single definition for both classes**. Today
  `BackgroundScorer.score` refits IForest/LOF/MCD on positives and scores both
  classes (consistent); the docstring also allows the `anomalias_a.csv` path for
  positives. Decide the final definition and make the methods note match the code
  (never mix paths per class).
- Re-run the ablation; record either **promoted** (then add to
  `EVERYDAY_FEATURES`/candidate promotion path and re-baseline) or **not
  promoted** (then the "exist but unproven" wording stays, this time on a frozen
  pool).
- **Guard against gate-shopping**: the gate is fixed by §C.2 (mean Δ > margin,
  CI excludes 0, not negative in any protocol); report every variant tried.

**Deliverable**: encoding decision recorded + final ablation verdict + methods
note updated.

### WS10 — Model set & tuning

- Extend `model_candidates` (`src/wildfire/analysis/occurrence_model.py:69`):
  class-balanced variants (e.g. `class_weight="balanced"`), and optionally a
  regularized logistic second floor; keep sklearn-only defaults, fresh per fold.
- Decide the tuning budget. Default: **small hand grid** fit inside the train
  fold (seed 42, no new deps). optuna remains [OPEN] (new dependency, needs
  approval).
- Re-run the model-comparison cell; keep the floor model for permutation
  importance and pick the reported model by mean PR-AUC (min reported too).

**Deliverable**: final model table + a recorded decision on the model
set/tuning.

### WS11 — Deployment unit & calibration (Plan D)

- Implement the **cheap** deliverable now: an empirical **CCAA-day fire-rate
  table** in notebook 04's output cell (FIRMS counts against the full CCAA×day
  grid; no extra quota), reported alongside the model scores.
- Keep the scoring pipeline separable so a calibration stage can be added later.
- Record the decision in the methods note: **ranking-only** default; CCAA-day
  probability as the actionable upgrade; point-level calibration not
  recommended; hazard/rate reframing as future work.

**Deliverable**: fire-rate table in 04 + the deployment-unit decision in the
methods note.

### WS12 — Docs & housekeeping (finish WS6)

- `AGENTS.md`: document `src/wildfire/analysis/` (`occurrence_model.py`,
  `ablation.py`), the notebook-04 outline, and the background chain entry
  (05 stage feeding only 04).
- `TODO.md`: PCA provenance (closed by WS8), calibration follow-up, larger pool.
- Update the status doc with the phase-2 re-run.

**Deliverable**: docs match reality; suite green.

## Sequencing

1. **WS8** first — it can invalidate WS9's `cluster_meteo` result and is pure code.
2. **WS7** runs in parallel (team data) — phase-2 re-runs wait on the freeze.
3. **WS9** after WS8; **WS10** is largely independent and can run on the current
   pool.
4. **WS11** independent (no pool dependency).
5. **WS12** last; final re-run of 04 once the pool is frozen.

## Verification

- `pytest tests/` stays green (550 today + the WS8/WS9/WS10 tests).
- `ruff check src/ tests/` and `mypy src/` at their pre-existing baselines — no
  new errors.
- Notebook 04 executes headless (`nbclient`) end-to-end on the frozen pool; every
  run states pool size, ratio, prevalence and per-fold min.
- WS8 provenance: PC1–7 correlation with `features_a.csv` ≈ 1 (up to sign),
  `n_components == 7`, cumulative variance ≈ 0.915.

## Residual risks

- **Unfrozen pool** — the frame changes while the fill runs; phase-2 numbers are
  comparable only at freeze.
- **PCA sign/order indeterminacy** — compare components up to sign before
  declaring a mismatch.
- **Gate-shopping** — trying many candidate variants can promote a fluke; the
  gate is fixed and every variant must be reported.
- **New dependencies** — optuna/xgboost/lightgbm remain [OPEN] and out of scope
  unless approved.
- **Small / single-class folds** — low-positive CCAAs and exterior quarters;
  `rank_metrics` needs both labels; keep per-fold reporting.
- **Encoding leakage** — one-hot/posteriors must stay fold-fit
  (`categorical_to_dummies` fits on train); posteriors are continuous but their
  GMM is fit on positives only.

## Open items carried

- [OPEN] `score_anomalia` exact formula (final definition decided in WS9).
- [OPEN] Candidate model set + tuning budget (WS10 default: small hand grid).
- [OPEN] `cluster_meteo` categorical vs posteriors (WS9 decides by the gate).
- [OPEN] PCA provenance (WS8).
- [OPEN] Deployment unit & calibration route (WS11 default: ranking-only +
  CCAA-day rates).
- [OPEN] git-tracking policy for large generated CSVs.
