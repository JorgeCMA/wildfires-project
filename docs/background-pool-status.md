# Background pool status for the demo (final, 2026-10-25)

The supervised occurrence model (block 04) is demo-ready and its **negatives** —
the background pool `data/processed/expanded/background_days.csv` — are complete:
the synthetic weather fill reached **100,000 / 100,000** rows and notebook 02d
was rebuilt from it, leaving **97,630** usable background rows (2023: 48,622;
2024: 49,008). Every year clears the configured `background_ratio: 2` target
(3.31× and 5.80×), so the pool is **frozen as of 2026-10-25** and every
model-derived number in the current artifacts is final for this snapshot.

Check the live number any time with:

```powershell
python scripts/background_pool_status.py
```

The demo app (`pip install -e ".[demo]"` then `streamlit run demo/app.py`) reads
only the notebook 01/04 artifacts and reports the same numbers in its *Estado*
tab; its Plan-C **live** scorer (coordinates + Open-Meteo) is phase 2.

| year | positives | needed (ratio 2) | have | achieved |
|---|---|---|---|---|
| 2023 | 14,694 | 29,388 | 48,622 | 3.31 |
| 2024 | 8,448 | 16,896 | 49,008 | 5.80 |
| total | 23,142 | 46,284 | 97,630 | 2.11 |

## Frozen snapshot (2026-10-25)

- Notebook 02d rebuilt from the complete weather file: 97,637 rows → 97,630
  after the same-day leakage filter (7 rows dropped).
- Notebook 04 retrained with `sample_background` at ratio 2 (46,284 negatives;
  prevalence 0.328). Best model: `hist_gradient_boosting`, mean PR-AUC 0.7760 /
  ROC-AUC 0.8908 across 25 folds; the candidate set is **not promoted**
  (Δ +0.0073, CI95% [−0.0001, +0.0163]) and neither are the trivial controls.
- Route-C calibration (margins 5/10/20 km): measured base rates
  0.1229 % / 0.4640 % / 1.7095 %; prior-corrected means
  0.2284 % / 0.8485 % / 2.9985 %; ECE 0.0271 / 0.0247 / 0.0229.
- Artifacts refreshed: `occurrence_scores.csv` (138,852 rows),
  `occurrence_fold_metrics.csv`, `ccaa_day_fire_rates.csv` (18 CCAA) and
  `calibration_reliability.csv`; the demo bundle was regenerated with
  `scripts/fit_occurrence_model.py` (HGB, train 2023 44,082 / calibrate 2024
  25,344 rows; margins 5/10/20 km).

## What remains caveated (not a pool caveat)

- **Not a point probability.** The contract is a **ranking**; the route-C
  exceedance probability is a wide-margin (5–10–20 km) approximation, not a
  point fire probability.
- **Not field accuracy.** PR-AUC is inflated by the ~1:2 framing; quote it as a
  ranking metric.
- **Relative bands.** `risk_band` / `risk_percentile` are relative to the scored
  batch by design; `ccaa_fire_rate` comes from the detections, not the pool.
- **PCA provenance**, the **2024-Q1 fold audit**, the background-leakage
  contract and the pipeline method are unchanged.

## Rebuild runbook (reproducibility)

1. Background weather: `scripts/enrich_weather_batch.py` with
   `--csv-path data/processed/synthetic/firms_spain_background_weather_partial.csv`
   and `--complete-path data/processed/synthetic/firms_spain_background_weather.csv`
   (spends Open-Meteo quota).
2. `notebooks/02d_background_data.ipynb` → `background_days.csv`.
3. `python scripts/background_pool_status.py` → `READY`.
4. `notebooks/04_aprendizaje_supervisado.ipynb` and `pytest -m experiment -s`.
5. `python scripts/fit_occurrence_model.py` (demo bundle, gitignored).
