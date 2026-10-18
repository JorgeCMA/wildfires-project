# Provisional status for the demo (2026-10-18)

The supervised occurrence model (block 04) is demo-ready, but its **negatives**
— the background pool `data/processed/expanded/background_days.csv` — are still
being filled in increments. Until the pool reaches the configured ratio, every
*model-derived* number is **provisional and may not be final**. The full-data
run is expected for the real demo once the pool is filled.

**Today:** pool **23,895** rows vs a ratio-2 target of **46,284** (~1.03:1 used;
`sample_background` falls back to "use the whole pool"). Weather is filled for
**24,500 / 100,000** skeleton rows. Check the live number any time with:

```powershell
python scripts/background_pool_status.py
```

| year | positives | needed (ratio 2) | have | achieved |
|---|---|---|---|---|
| 2023 | 14,694 | 29,388 | 11,980 | 0.82 |
| 2024 | 8,448 | 16,896 | 11,915 | 1.41 |
| total | 23,142 | 46,284 | 23,895 | 1.03 |

## What is provisional (moves when the pool fills)

- **PR-AUC / ROC-AUC** and the **candidate-model ranking**.
- **Ablation verdicts** (`ablation_verdict`): whether a candidate is promoted.
- **Route-C calibration** (`calibration_reliability.csv`): the measured base
  rates and reliabilities depend on the background sample.
- The **pool ratio itself** (1.03 vs the configured 2).

## What is stable (not a data caveat)

- **`ccaa_fire_rate`** — the empirical CCAA×day rate comes from the **detections**,
  not the background, so it is unaffected by the fill.
- **`risk_band` / `risk_percentile`** — *relative to the scored batch* by design;
  they are not an absolute danger scale regardless of pool size.
- **PCA provenance**, the **2024-Q1 fold-audit verdict**, the background-leakage
  contract and the pipeline method.

## What not to present as

- A **probability** of fire at a point. The contract is a **ranking**; the route-C
  exceedance probability is provisional and only honest at **wide margins**
  (5–10–20 km).
- **Real rare-event performance.** PR-AUC is inflated by the ~1:2 framing; quote
  it as a ranking metric, not field accuracy.
- A **final** result: state "provisional pending the full background pool".

## When the pool is filled (runbook)

1. Resume the background weather (slow, resumable; spends Open-Meteo quota —
   pass `--complete-path` explicitly, the default points at the *positives* file):

   ```powershell
   python scripts/enrich_weather_batch.py `
     --csv-path data/processed/synthetic/firms_spain_background_weather_partial.csv `
     --complete-path data/processed/synthetic/firms_spain_background_weather.csv `
     --max-batches 20
   ```

2. Re-run `notebooks/02d_background_data.ipynb` to rebuild `background_days.csv`.
3. Confirm readiness: `python scripts/background_pool_status.py` → `READY`.
4. Re-run the model + experiments:
   `notebooks/04_aprendizaje_supervisado.ipynb` and `pytest -m experiment -s`.
5. Drop the "provisional" wording in `TODO.md`, the nb04 methods note and this
   file; refresh `occurrence_scores.csv`, `ccaa_day_fire_rates.csv` and
   `calibration_reliability.csv`.
