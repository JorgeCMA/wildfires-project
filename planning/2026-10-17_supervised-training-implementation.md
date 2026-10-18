# Supervised training — implementation plan (Plans B–D of `plan_supervised_implementation.md`)

**Date**: 2026-10-17. **Status**: ready to implement; Plan A data is consumed via
`load_background_weather()` (implemented, tested) while the team fills the partial
on the side.

**Revision 2026-10-18**: Steps 1–2 are done ✅; Steps 3–6 are refined in
`planning/2026-10-18_supervised-training-implementation.md` (decisions of
2026-10-18: ratio 2 fallback-to-all, per-file BLH median, leakage re-check,
candidates produced in 04, fold-fit one-hot encoding, margin + bootstrap-CI
ablation gate, PCA provenance TODO). That file is the live plan for Steps 3–6.

Everything here implements the contracts already decided in
`planning/plan_supervised_implementation.md` (referred to below as **PLAN**).
The occurrence contract (`occurrence_features`, `assemble_modeling_frame`,
`background_skeleton`, `blocked_splits`, `rank_metrics`) already exists in
`src/wildfire/processing/occurrence.py` with 23 tests — this plan builds on top
of it, it does not redesign it.

## Already in place (do not rebuild)

- `src/wildfire/processing/occurrence.py` — everyday allowlist, satellite
  blocklist, wind circular expansion, season tag, blocked splits, PR-AUC.
- `load_background_weather()` (`src/wildfire/enrichment/weather_batch.py`) —
  final→partial fallback, temperature_2m-filled rows only.
- Config knobs: `modeling.occurrence` (ratio, seed, cv groups).
- `notebooks/04_aprendizaje_supervisado.ipynb` — two-cell skeleton, to be
  replaced per PLAN §C.3.

## Step 1 — Persist k12 typology labels (PLAN §B.2.1) ✅ 2026-10-18

**What**: `cluster_labels.csv` on disk only has k5; 03e refits k12 inline
(notebook 03e cells 3/17). Extracted into code and persisted.

**Correction vs PLAN §B.2.1 (verified against the notebook)**: the elected
k12 uses **KMeans hard labels** on `StandardScaler(PC1–7)`; the
full-covariance GMM k=12 provides the **posteriors** (`gmm_k12_full_p{j}`),
it is not the hard-label source. PLAN said "GMM k12" — imprecise.

- `src/wildfire/processing/typology.py`: `fit_k12_typology(features)` +
  `K12Typology` dataclass (fitted scaler/KMeans/GMM). `labels_for` reproduces
  the 03e labels; `score_background` is the same method documenting the
  contract: background is scored by fitted-on-positives objects, never refit.
- `scripts/persist_k12_labels.py`: fits on `features_a.csv` (positives only),
  writes `data/processed/unsupervised/cluster_labels_k12.csv` (23,142×14:
  `detection_id`, `cluster_meteo`, 12 posteriors).
- Tests (`tests/test_k12_labels.py`, 10): determinism under seed 42,
  per-`detection_id` keying (shuffled input → same labels), posterior rows
  sum to 1, background scoring via fitted objects, and a real-data test
  asserting ARI = 1.0 against the 03e inline KMeans k12 refit.
- Exported from `wildfire.processing` (`K12Typology`, `fit_k12_typology`,
  `K12_ELECTED_K`, `K12_PC_COLUMNS`).

**Deliverable**: ✅ `data/processed/unsupervised/cluster_labels_k12.csv` on disk.

## Step 2 — Score background rows with fitted-on-positives objects (PLAN §B.2.2 + §C.4) ✅ 2026-10-18

**What**: background needs `cluster_meteo` from the same fitted objects, and
a continuous `score_anomalia` (PLAN §B.2.3 default: row-wise mean of the
three z-scored `*_score` columns, min-max to [0,1]).

- `src/wildfire/processing/reintegration.py` (new):
  - `load_unsupervised_scores(anomaly_set="a")` — persisted 03c scores,
    keyed by `detection_id` (never by row order).
  - `score_anomalia(df)` — the documented default formula (descriptive on
    the frame passed; reproduces the positives' score from `anomalias_a.csv`).
  - `fit_background_scorer(positives) -> BackgroundScorer` — fits on
    positives only: weather-PCA pipeline (raw weather → PC1–7, set A),
    `K12Typology` (Step 1), IForest + LOF (novelty mode) + MinCovDet on the
    PC scores, and the `score_anomalia` normalization (z-score + min-max
    params from positives). `.score(df)` labels/scores positives
    (`detection_id`) or background (`background_id`) with those fitted
    objects — never refits; adding background rows never changes other
    rows' scores (tested).
- Contract extension in `occurrence.py`: `CANDIDATE_FEATURES =
  ["cluster_meteo", "score_anomalia"]`; `assemble_modeling_frame(...,
  pos_candidates=, neg_candidates=)` appends them after the everyday block
  in canonical order (satellite check still applies; frames must come in
  pairs, lengths enforced). Until the ablation gate passes they stay out of
  `EVERYDAY_FEATURES` — "exist but unproven" in the methods note.
- Tests (`tests/test_reintegration.py`, 16): formula direction/range, seed
  determinism, score stability under added background rows, id keying for
  both classes, error cases, and the full candidate-frame contract
  (canonical order, pair/length/satellite validation). The 23 occurrence
  tests stay green.

**Deliverable**: ✅ background (and positives) rows carry `cluster_meteo` +
`score_anomalia` via one call; ablation-ready modeling frame.

## Step 3 — Training units in `src/`, training loop in notebook 04 (PLAN §C.1)

**Architecture rule (user decision, 2026-10-18)**: same structure as the rest
of the project — the notebook performs the calls: blocked-split iteration,
scaler-fit-per-fold, `model.fit`, scoring, candidate comparison. `src/`
provides only the small tested units the cells call (as 03b/03e do with
`fit_gmm_range`/`fit_labels`), never the experiment loop.

- `src/wildfire/analysis/occurrence_model.py` (new):
  - `model_candidates(seed=42) -> dict[str, estimator]` — the provisional
    three (`LogisticRegression` floor, `RandomForestClassifier`,
    `HistGradientBoostingClassifier`), sklearn only, sensible defaults, no
    tuning budget (optuna is OPEN). Fresh instance per name per fold.
  - `fold_scaler_fit(X_train) / fold_scaler_apply(scaler, X)` — explicit
    helpers so the per-fold scaler fit (hard rule 4.3) is one visible call
    in the cell, not hidden inside a loop body elsewhere.
  - `categorical_to_dummies(frame, cols, train_columns=None)` — one-hot
    encoding **fit on the train fold** (`cluster_meteo` enters as
    categorical; PLAN default), aligned across folds via `train_columns`.
- The notebook cells do: `for protocol in ("season", "ccaa"): for split in
  blocked_splits(...): ... fit ... rank_metrics(...)` and collect the
  fold-metrics frame themselves; per-CCAA/per-season tables and PR curves
  from that frame.
- No calibrated probabilities anywhere (PLAN §D); scores are rankings.

**Deliverable**: notebook cells produce the tidy fold-metrics frame; src
units tested (`tests/test_occurrence_model.py`).

## Step 4 — Ablation (PLAN §C.2, the promotion gate)

Fits run in notebook cells (Step 3 pattern, with/without the candidate
columns); `src` provides the pure, tested verdict unit:

- `src/wildfire/analysis/ablation.py` (new):
  `ablation_verdict(with_frame, without_frame) -> dict` — takes the two
  per-fold PR-AUC frames collected by the notebook and applies the
  promotion rule: mean ΔPR-AUC > 0 and not negative in any protocol,
  plus the `month` / `ccaa` one-hot negative controls (same function on
  their frames).
- Tests: on synthetic fold frames the verdict flips correctly when a
  separable candidate column is present/absent (PLAN §C.4).

**Deliverable**: verdict computed by code from notebook-collected frames.

## Step 5 — Notebook 04 (PLAN §C.3, one cell per step)

Replace the skeleton, orchestrating only:

0. Title/objectives (bilingual), rule reminder (blocked splits, seed 42, raw
   units, no satellite columns).
1. Load positives (`firms_spain_final_lags_index.csv`) + background via
   `load_background_weather()`; BLH median imputation exactly as notebook 01;
   assemble frame; sanity cells (label balance 1:2, per-CCAA/per-season
   counts, no-NaN policy).
2. Baseline fits + blocked-split evaluation (season, ccaa) → fold table.
3. Candidate model comparison table.
4. Reintegration: k12 labels, background scoring, candidate columns.
5. Ablation + verdict cell (promoted / not promoted).
6. Outputs: `data/processed/predictions/occurrence_scores.csv`
   (id, label, score, held-out group), fold-metrics summary CSV, PR curves,
   per-CCAA PR-AUC choropleth, permutation importance for the floor model,
   package-versions cell, methods-note paragraph (ranking not probability,
   background sampling assumptions, empirical CCAA-day fire rates per
   PLAN §D recommendation).

## Step 6 — Tests & housekeeping

- New tests land next to their module (Steps 1–4); full suite must stay green
  (`pytest tests/`).
- Update `AGENTS.md` (new modules under `src/wildfire/analysis/`, notebook 04
  outline) once the code lands.
- Keep `ruff check` / `mypy src/` at their current baselines — no new errors.

## Execution order & parallelization

1 → 2 can proceed **now** on the partial background (loader fallback), since
they only need filled weather rows. 3 → 4 are pure code, no data dependency.
5 waits for 1–4 and needs "enough" background rows to be meaningful (the
loader keeps working as the fill progresses — metrics will simply improve).
Step 2's background scorer re-runs cheaply, so no rework when the fill
completes.

## Open items carried from PLAN §8 (unchanged)

- `score_anomalia` exact formula (Step 2 implements the documented default).
- Candidate model set + tuning budget (Step 3 ships the provisional three).
- `cluster_meteo` as categorical vs posteriors (default: categorical,
  one-hot inside fold).
- Deployment unit (PLAN §D): ranking-only default; CCAA-day rates reported
  alongside scores in the notebook deliverables cell.
