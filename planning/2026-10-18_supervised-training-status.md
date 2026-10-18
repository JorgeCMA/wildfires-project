# Supervised training — status & next steps (after WS1–WS4)

**Date**: 2026-10-18. **Status**: WS1–WS4 implemented and green; notebook 04
runs headless end-to-end on real data. This file is the running status derived
from `planning/2026-10-18_supervised-training-implementation.md` (the plan) and
`planning/plan_supervised_implementation.md` (the strategy). Commit:
`8f6f10f` — "Implement supervised occurrence pipeline (WS1-WS4) + raw background
artifact".

**Objective (unchanged)**: a *risk ranking* of fire occurrence from everyday
(non-satellite) conditions. Positives = FIRMS detections
(`data/processed/expanded/firms_spain_final_lags_index.csv`, raw units, key
`detection_id`); negatives = enriched background CCAA×day rows
(`data/processed/expanded/background_days.csv`, raw units, key
`background_id`). Primary metric **PR-AUC**, secondary ROC-AUC; **no calibrated
probabilities**.

---

## TL;DR

- All four modeling workstreams are coded, tested and executed.
- The enriched background artifact was **normalized and mis-keyed** as delivered;
  `notebooks/02d_background_data.ipynb` was fixed and **regenerated in raw units**
  (key `background_id`, satellite columns dropped, leakage rows dropped). A
  descriptive z-scored copy (`background_days_normalized.csv`) is EDA-only.
- `pytest tests/` = **550 passed, 2 deselected**; touched files are ruff/mypy clean.
- Notebook 04: best provisional model is `hist_gradient_boosting`; the
  reintegration candidates are **not promoted** (ΔPR-AUC +0.0039, CI95%
  [−0.0033, +0.0116]); negative controls correctly stay unpromoted.

---

## What is done

| WS | Scope | Where | Status |
|----|-------|-------|--------|
| WS1 | Consumption layer | `src/wildfire/processing/occurrence.py`, `tests/test_occurrence.py` | ✅ |
| WS2 | Model units | `src/wildfire/analysis/{__init__,occurrence_model}.py`, `tests/test_occurrence_model.py` | ✅ |
| WS3 | Ablation gate | `src/wildfire/analysis/ablation.py`, `tests/test_ablation.py` | ✅ |
| WS4 | Notebook 04 | `notebooks/04_aprendizaje_supervisado.ipynb` | ✅ |
| WS5 | Reintegration | `src/wildfire/processing/reintegration.py`, `scripts/persist_k12_labels.py`, `tests/test_reintegration.py`, `tests/test_k12_labels.py` | ✅ (PCA provenance [OPEN]) |
| WS6 | Docs / config | `configs/project.yaml`, `AGENTS.md`, `TODO.md` | ◑ partial |
| — | Background artifact | `notebooks/02d_background_data.ipynb`, `data/processed/expanded/background_days.csv` | ✅ |

### WS1 — consumption layer

- `load_background_days(path=None)` — config `output.background_days`; asserts
  `background_id`.
- `validate_background_days(background, positives, leakage_max_km=1.0)` —
  fail-closed: required columns, no satellite columns (`rain`/`is_day`/`type==0`
  allowed), unique key, per-column NaN report, same-day/≤ `leakage_max_km`
  re-check.
- `drop_leaking_background(background, positives, leakage_max_km=1.0)` — **new**;
  drops the leakage rows so a rebuilt artifact passes the contract as-is.
- `sample_background(background, n_positives, *, ratio, seed)` — seeded, sorted
  by `background_id`; returns the whole pool when the target is unreachable.
- `occurrence_settings()` exposes `background_ratio`, `cv_groups`,
  `leakage_max_km`, `random_state`.

### WS2 — model units

- `model_candidates(seed=42)` → `logistic`, `random_forest`,
  `hist_gradient_boosting`.
- `fold_scaler_fit` / `fold_scaler_apply` (continuous columns only),
  `categorical_to_dummies` (train-fold fit, aligned, unseen → all-zero),
  and the continuous/categorical column policy derived from `EVERYDAY_FEATURES`.

### WS3 — ablation gate

- `ablation_verdict(with_metrics, without_metrics, margin=0.0, seed=42, n_boot=2000)`
  → per-fold ΔPR-AUC, mean, bootstrap CI, per-protocol min, `promoted`.
- `bootstrap_mean_ci`. Negative controls run through the same function.

### WS4 — notebook 04

15 cells (bilingual markdown + single-imports cell): load/validate/sample/assemble
→ blocked baseline (PR-AUC = fold prevalence) → candidate comparison →
reintegration (`fit_background_scorer`, positives-only fit) → ablation + negative
controls → outputs (OOF predictions CSV, fold-metrics CSV, PR curves, per-CCAA
choropleth, permutation importance, package versions) → methods note.

### Background artifact (the fix that unblocked WS4)

`02d_background_data.ipynb` now:

1. keeps columns raw (same units as `firms_spain_final_lags_index.csv`);
2. renames the key `detection_id` → `background_id`;
3. drops the 6 all-null satellite columns;
4. drops same-day leakage rows via `drop_leaking_background`;
5. saves raw to `background_days.csv` and a **descriptive-only** z-scored copy to
   `background_days_normalized.csv` (`background_id`/`riesgo_antropico_alto` not
   scaled).

Regenerated artifact: **23,895 × 58**, key `background_id` (int64, unique),
no satellite columns, no NaN, `validate_background_days` → 0 duplicates, 0 leaks.

---

## Verified results (run of 2026-10-18, pool not yet frozen)

- Background: 23,895 rows validated (0 dup, 0 leak, no NaN).
- Sampling: requested `ratio 2` = 46,284; **pool smaller → all 23,895 used**
  (effective ≈ 1.03:1).
- Baseline PR-AUC (= mean fold prevalence): **0.456**.
- Best provisional model by mean PR-AUC: **`hist_gradient_boosting`**.
- Ablation of `cluster_meteo` + `score_anomalia`: **not promoted**
  (Δ +0.0039, CI95% [−0.0033, +0.0116]); controls `month`/`ccaa` not promoted
  (−0.0009 / +0.0000).
- Outputs written: `data/processed/predictions/occurrence_scores.csv`
  (94,074 OOF rows), `data/processed/predictions/occurrence_fold_metrics.csv`.

### Verification commands

```bash
pytest tests/                       # 550 passed, 2 deselected
ruff check src/ tests/ scripts/     # pre-existing baseline only, no new errors
mypy src/wildfire/processing/occurrence.py
# notebook 04 headless end-to-end (nbclient, cwd=notebooks/)
```

---

## What is next

1. **Freeze / grow the pool, then re-report.** The metrics above are on an
   unfrozen pool; re-run once the fill reaches `background_ratio: 2` (or is
   frozen) and report the baseline PR-AUC + per-fold min alongside every number.
2. **PCA provenance [OPEN]** — verify `fit_weather_pca` reproduces the
   `features_a.csv` / 03b space before trusting `cluster_meteo` values; otherwise
   the candidate is scored in a different space.
3. **WS6 remainder** — finish the AGENTS notebook-04 outline and add a PCA
   provenance entry to TODO.
4. **Decide the encoding of `cluster_meteo`** — one-hot categorical (default) vs
   the 12 GMM posteriors.
5. **Candidate model set / tuning budget** — currently the provisional three with
   no tuning; decide whether to tune.
6. **Deployment unit & calibration (Plan D)** — ranking-only default; CCAA-day
   empirical rate as the actionable upgrade (ideally re-derive the candidates if
   the ablation stays negative).

## Open decisions carried

- [OPEN] `cluster_meteo` categorical (12 levels) vs 12 posteriors.
- [OPEN] candidate model set + tuning budget.
- [OPEN] deployment unit / calibration route (Plan D).
- [OPEN] PCA provenance (blocker for trusting `cluster_meteo`).
- [OPEN] git-tracking policy for large generated CSVs (repo currently tracks
  them, incl. `background_days*.csv` and `predictions/*.csv`).

## Residual risks

- **Unfrozen pool** — the exact modeling frame changes while the background fill
  runs; results are comparable only once frozen.
- **Small / single-class folds** — low-positive CCAAs and exterior quarters;
  guards in WS1/WS4 + per-fold reporting.
- **Leakage** — re-verified on the modeling side (never trusted from the build).
- **Encoding** — one-hot must stay fold-fit or it leaks across folds.
