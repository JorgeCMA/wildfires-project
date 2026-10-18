# Route C — wide-margin exceedance probability (provisional)

Third deployment track after `docs/deployment-unit-A-B.md`. The occurrence model
is a **ranking**, not a probability (prevalence is set by the negative sampling).
Route C asks the only calibratable question: not "P(fire at this exact point)"
(near-zero, unestimable) but

> **P(≥1 detection within `R` km of a point on day `d`)** — a wide-margin
> *exceedance* probability.

Code: `src/wildfire/analysis/calibration.py`. Driver: `notebooks/04`
section 8. This is **experimental** (see Limits).

## Method

1. **Temporal split, no shuffle** — train the model on **2023**, calibrate on
   **2024**. Same per-fold one-hot/scaler rule as the rest of block 04
   (encoding fit on the train year only).
2. **Margin labeling** — a 2024 validation point is *positive-at-R* if a
   same-day detection lies within `R` km (`margin_positive_labels`, haversine,
   date-partitioned `BallTree`). Detections are positives by definition.
3. **Isotonic map** — `IsotonicRegression` maps the model score to a
   probability; the persisted calibrator is fit on all of 2024.
4. **Cross-fitted reliability** — reported reliability/Brier come from
   `cross_fitted_isotonic` (2024 days split into 5 contiguous blocks; each
   block predicted by a calibrator fit on the others), so the curve is **not**
   scored on the calibrator's own fit.
5. **Prior correction** — `prior_correct` rescales from the *sampled*
   prevalence of the 2024 frame to the **measured base rate**
   (`exceedance_base_rate`), estimated directly on the **full background pool**
   (model-free), with a day-block bootstrap CI.

Margins swept: **R ∈ {5, 10, 20} km** (`modeling.occurrence.calibration_margins_km`).
`eventos.csv` gives the effective event count (1,178 unique events behind the
23,142 detections).

## Results (2023 → 2024, HGB)

Train 2023 = 26,674 rows; calibrate 2024 = 20,363 rows (`hist_gradient_boosting`).

| R (km) | n | n margin+ | sampled prev. | **measured base rate** | prior-corrected mean | Brier | log-loss | ECE |
|---|---|---|---|---|---|---|---|---|
| 5 | 20,363 | 8,466 | 0.416 | **0.00130** | 0.00255 | 0.160 | 0.475 | 0.0276 |
| 10 | 20,363 | 8,511 | 0.418 | **0.00502** | 0.00958 | 0.161 | 0.477 | 0.0257 |
| 20 | 20,363 | 8,671 | 0.426 | **0.01791** | 0.03300 | 0.161 | 0.480 | 0.0255 |

Reading:

- **Base rate scales with area** — 0.13 % → 0.50 % → 1.79 % per point-day as
  the margin grows 5→20 km (roughly linear in area for nearby fire clusters).
  These are the *absolute* chances, measured from data, no model involved.
- **`sampled prev.` ≈ 0.42**, not 0.33: the frame is 1:2 (33 % positives, all
  margin-positive) *plus* the background rows that are themselves margin-positive.
  It is the artificial training prevalence the contract warns about, and exactly
  why the raw score must not be read as a probability.
- **Prior-corrected mean** (0.26 %, 0.96 %, 3.3 %) is the deployment-scale
  number: the average exceedance probability once rescaled to the real base rate.
- **ECE ≈ 2.6 pts** on the sampled scale after isotonic → the ranking's shape is
  already close to a probability; the isotonic map mostly fixes the level.
- **Brier ≈ 0.16** is dominated by the sampled prevalence (~0.42), not by the
  model, so it is *not* comparable to a real-world Brier.

## Output — `calibration_reliability.csv`

30 rows (3 margins × 10 quantile bins), columns
`bin_lower, bin_upper, mean_predicted, observed_freq, n, margin_km` — the
reliability curves behind the plot in notebook section 8.

## Reproduce

```bash
python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('notebooks/04_aprendizaje_supervisado.ipynb', as_version=4); \
NotebookClient(nb, timeout=2400, resources={'metadata':{'path':'notebooks'}}).execute(); \
nbformat.write(nb, 'notebooks/04_aprendizaje_supervisado.ipynb')"

pytest tests/test_calibration.py
```

## Limits (why this stays provisional)

- **Pool not frozen.** The background pool is below `background_ratio: 2`, so
  the base rates and every calibration number can move when it grows.
- **Sampled, not uniform, evaluation.** The 2024 frame is positives + a
  CCAA×day background sample, not a uniform point grid; the prior correction
  targets the *background-population* margin rate, which is the best available
  measured base rate, not a census.
- **Not a point probability.** It is honest only at wide area margins; do not
  read it as "P(this spot burns)".
- **No live scorer / persisted model.** These are historical artifacts; the
  production deliverable is still the **ranking** (`occurrence_scores.csv`, A+B).
