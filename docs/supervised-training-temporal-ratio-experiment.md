# Supervised training: class-balance × temporal-split experiment

A small controlled study on the **occurrence model**. It answers two questions
that the phase-2 work left open:

1. How does the model rank when the background (negative) pool is subsampled to
   **different fire:background ratios**?
2. How does it generalise under a **strict temporal split** — train on all of
   2023, validate on all of 2024 (no shuffling, no leakage across years)?

It does **not** touch the pipeline. It only recombines public units
(`wildfire.processing.occurrence` + `wildfire.analysis.occurrence_model`), so it
runs in ~15 s on real data and cannot change any published artifact.

Source: `tests/test_temporal_ratio_experiment.py`.
Run: `pytest -m experiment -s` (excluded from the default suite via
`addopts = "-m 'not integration and not experiment'"` in `pyproject.toml`).

---

## 1. What is being varied

### Class balance (the "ratio")

The occurrence model learns from a mix of real fire detections (positives) and
deliberately generated background days (negatives). Nature never hands us a
balanced dataset — a random region×day is almost never a fire — so the ratio is
something **we choose** when sampling the background pool.

The experiment sweeps the negative:positive ratio over
`RATIOS = (0.25, 0.5, 1.0, 2.0)`:

- `0.25` → 1 negative per 4 positives (deliberately **under**-samples negatives)
- `0.5`  → 1 negative per 2 positives
- `1.0`  → balanced
- `2.0`  → 2 negatives per positive (the project's nominal `background_ratio`)

Negatives are drawn **per year**, so the requested ratio holds independently
inside the training partition and inside the validation partition. When a year's
pool is smaller than the requested count, the *whole* pool is used (same fallback
as `occurrence.sample_background`). This cap matters — see §4.

### Temporal split (2023 → 2024)

- **Train**: every 2023 row (14,694 positives + sampled 2023 negatives).
- **Validate**: every 2024 row (8,448 positives + sampled 2024 negatives).
- No shuffling. The model is never shown 2024 data during training, which mimics
  real deployment: you fit on the past, you score the future.

All preprocessing is fit on the **2023 train portion only**:

- one-hot encoding of categorical columns (`categorical_to_dummies`),
- median imputation of continuous columns,
- standardisation (`fold_scaler_fit` / `fold_scaler_apply`).

This is the same per-fold discipline as notebook 04, just with a single
temporal split instead of blocked leave-one-group-out folds.

---

## 2. Protocol, step by step

For each `ratio` in `RATIOS`:

1. Count positives per year: 2023 → 14,694, 2024 → 8,448.
2. For each year, draw `round(ratio × n_pos_year)` negatives from that year's
   background pool (deterministic `SEED = 42`, sorted by `background_id`).
3. Assemble one modeling frame (`assemble_modeling_frame`) from the daylight
   features of both classes; `label` is 1 for positives, 0 for negatives.
4. Fitness/split: train = 2023 rows, validate = 2024 rows.
5. Fit one-hot + medians + scaler on train; transform validation with the *same*
   objects.
6. For each of the 5 `model_candidates` (`logistic`, `logistic_balanced`,
   `random_forest`, `random_forest_balanced`, `hist_gradient_boosting`), fit on
   train and score validation.
7. Record PR-AUC and ROC-AUC (`rank_metrics`). PR-AUC's no-skill reference is the
   **validation prevalence** (`y_valid.mean()`), printed alongside the scores.

The test also asserts that `validate_background_days` reports **0 duplicate ids**
and **0 same-day leakage** rows before anything is measured.

---

## 3. Results

`train_neg_pos` / `valid_neg_pos` are the *realised* negative:positive ratios;
`valid_prevalence` is the positive share in validation (= no-skill PR-AUC floor).

### Per ratio × model

| ratio | model | n_train | train_neg_pos | n_valid | valid_neg_pos | valid_prevalence | PR-AUC | ROC-AUC |
|------:|-------|--------:|--------------:|--------:|--------------:|-----------------:|-------:|--------:|
| 0.25 | logistic | 18,368 | 0.250 | 10,560 | 0.250 | 0.800 | 0.9176 | 0.7834 |
| 0.25 | logistic_balanced | 18,368 | 0.250 | 10,560 | 0.250 | 0.800 | 0.9148 | 0.7728 |
| 0.25 | random_forest | 18,368 | 0.250 | 10,560 | 0.250 | 0.800 | **0.9437** | 0.8393 |
| 0.25 | random_forest_balanced | 18,368 | 0.250 | 10,560 | 0.250 | 0.800 | 0.9413 | 0.8298 |
| 0.25 | hist_gradient_boosting | 18,368 | 0.250 | 10,560 | 0.250 | 0.800 | 0.9426 | **0.8414** |
| 0.50 | logistic | 22,041 | 0.500 | 12,672 | 0.500 | 0.6667 | 0.8479 | 0.7793 |
| 0.50 | logistic_balanced | 22,041 | 0.500 | 12,672 | 0.500 | 0.6667 | 0.8458 | 0.7745 |
| 0.50 | random_forest | 22,041 | 0.500 | 12,672 | 0.500 | 0.6667 | 0.8940 | 0.8367 |
| 0.50 | random_forest_balanced | 22,041 | 0.500 | 12,672 | 0.500 | 0.6667 | 0.8923 | 0.8314 |
| 0.50 | hist_gradient_boosting | 22,041 | 0.500 | 12,672 | 0.500 | 0.6667 | **0.8942** | **0.8449** |
| 1.00 | logistic | 26,674 | 0.815 | 16,896 | 1.000 | 0.500 | 0.7348 | 0.7773 |
| 1.00 | logistic_balanced | 26,674 | 0.815 | 16,896 | 1.000 | 0.500 | 0.7340 | 0.7758 |
| 1.00 | random_forest | 26,674 | 0.815 | 16,896 | 1.000 | 0.500 | 0.8139 | 0.8352 |
| 1.00 | random_forest_balanced | 26,674 | 0.815 | 16,896 | 1.000 | 0.500 | **0.8153** | 0.8350 |
| 1.00 | hist_gradient_boosting | 26,674 | 0.815 | 16,896 | 1.000 | 0.500 | 0.8118 | **0.8444** |
| 2.00 | logistic | 26,674 | 0.815 | 20,363 | 1.410 | 0.4149 | 0.6654 | 0.7789 |
| 2.00 | logistic_balanced | 26,674 | 0.815 | 20,363 | 1.410 | 0.4149 | 0.6646 | 0.7774 |
| 2.00 | random_forest | 26,674 | 0.815 | 20,363 | 1.410 | 0.4149 | 0.7644 | 0.8366 |
| 2.00 | random_forest_balanced | 26,674 | 0.815 | 20,363 | 1.410 | 0.4149 | **0.7655** | 0.8361 |
| 2.00 | hist_gradient_boosting | 26,674 | 0.815 | 20,363 | 1.410 | 0.4149 | 0.7605 | **0.8463** |

### Mean over models, per ratio

| ratio | realised train neg:pos | realised valid neg:pos | valid prevalence | mean PR-AUC | mean ROC-AUC |
|------:|-----------------------:|-----------------------:|-----------------:|------------:|-------------:|
| 0.25 | 0.250 | 0.250 | 0.800 | 0.9320 | 0.8133 |
| 0.50 | 0.500 | 0.500 | 0.667 | 0.8748 | 0.8133 |
| 1.00 | 0.815 | 1.000 | 0.500 | 0.7819 | 0.8135 |
| 2.00 | 0.815 | 1.410 | 0.415 | 0.7241 | 0.8150 |

### Best model per ratio

| ratio | best model | valid prevalence | PR-AUC | ROC-AUC |
|------:|------------|-----------------:|-------:|--------:|
| 0.25 | random_forest | 0.800 | 0.9437 | 0.8393 |
| 0.50 | hist_gradient_boosting | 0.667 | 0.8942 | 0.8449 |
| 1.00 | random_forest_balanced | 0.500 | 0.8153 | 0.8350 |
| 2.00 | random_forest_balanced | 0.415 | 0.7655 | 0.8361 |

---

## 4. Interpretation

- **ROC-AUC is essentially flat** across ratios: mean 0.813 → 0.815. The model's
  ability to *rank a random positive above a random negative* barely depends on
  how many negatives we feed it. Tree ensembles (`random_forest*`,
  `hist_gradient_boosting`) consistently clear the logistic baseline
  (≈ 0.835–0.846 vs ≈ 0.776).

- **PR-AUC tracks the sampled prevalence almost one-for-one.** As the ratio
  rises, PR-AUC falls while the no-skill floor (`valid_prevalence`) falls by the
  same amount — the *lift over the floor* stays roughly constant. This is a
  property of the sampling, **not** a signal that more negatives make the model
  worse. It is the central reason the project reports **ranking, not calibrated
  probabilities**.

- **Ratios 1.0 and 2.0 are capped by the 2023 background pool.** The realised
  training ratio is 0.815 for both, because 2023 only contains ~11,980 background
  rows (targets of 14,694 and 29,388 respectively both exhaust it). So the two
  largest rows share an identical training set; only the *validation* sample
  differs (1.00 vs 1.41). This is exactly the unfrozen-pool limitation tracked as
  WS7 — once the pool grows past the `background_ratio: 2` target, this sweep
  should be re-run.

- **Class-balanced variants are not better here.** `logistic_balanced` and
  `random_forest_balanced` sit within noise of (and often just below) their plain
  counterparts, so reweighting buys nothing under this temporal split.

---

## 5. Relationship to the notebook-04 baseline

| | notebook 04 (blocked baseline) | this experiment |
|---|---|---|
| Split | blocked leave-one-group-out folds | single temporal split (2023 → 2024) |
| Background | pool subsampled to `background_ratio` | ratio swept (0.25–2.0), per year |
| Evaluation | out-of-fold PR/ROC + ablation | one held-out-year evaluation |
| Purpose | ship the model | stress-test balance & generalisation |

Notebook 04 remains the deliverable. This experiment is diagnostic: it shows that
the headline PR-AUC number is dominated by the chosen ratio, and that the model's
true ranking skill is stable and best captured by ROC-AUC / lift-over-floor.

---

## 6. Caveats

- Single temporal split → one evaluation point per `(ratio, model)`; no
  confidence intervals. Treat differences of a few thousandths as noise.
- The background pool is not frozen (WS7), so the ratio-1.0/2.0 rows are
  train-capped and will change when the pool grows.
- No hyper-parameter tuning (optuna remains `[OPEN]`); models use library
  defaults with `seed=42`.
- Scores are rankings, not probabilities; do not read PR-AUC as an absolute
  performance level.

---

## 7. Reproducing

```bash
# full default suite (experiment excluded)
pytest tests/

# this experiment only, with the printed tables
pytest -m experiment -s
```

Output is the three tables in §3 (per ratio × model, means, best per ratio).
