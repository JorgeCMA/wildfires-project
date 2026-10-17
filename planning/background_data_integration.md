# Background data — where it belongs in the project (2026-10-17)

**Context**: background days (random CCAA×day negatives, 1:2 against the 23,142
detections per `modeling.occurrence.background_ratio`) are being developed. This note
records the theoretical placement decision. Core principle: **background rows are a
modeling construct for the occurrence model, not observations** — they belong to the
supervised block and its supporting machinery, and nowhere else.

## Where it SHOULD be included

1. **`notebooks/04_aprendizaje_supervisado.ipynb`** — the only real consumer:
   negatives for `assemble_modeling_frame`, blocked-split evaluation (season/CCAA
   folds with both classes present), and the §7B ablation — background rows must be
   *scored by the same fitted objects* (scaler + GMM k12 for `cluster_meteo`,
   IF/LOF/Mahalanobis for `score_anomalia`), which only makes sense inside 04.

2. **`src/wildfire/processing/occurrence.py` + `tests/test_occurrence.py`** — the
   contract already anticipates it (`background_skeleton`, `assemble_modeling_frame`).
   To add: loader/validator for the finished background CSV, candidate-features path
   (`cluster_meteo`, `score_anomalia`), leakage-exclusion rule, and tests for all of
   it. Per project convention, fill/scoring logic >5 lines lives in `src/`.

3. **`configs/project.yaml`** — background CSV path, `n_per_day`, exclusion distance
   (default ~1 km of a same-day detection). No hardcoded paths/parameters.

4. **Docs** — AGENTS.md pipeline-chain entry (a stage feeding only 04), the
   supervised plan checklist, and the methods note on how negatives were built
   (sampling assumptions, quota-bounded weather, BLH imputation policy).

## Where it should NOT be included — and why

1. **Exploration notebooks (02a/02b/02c).** EDA describes *detections*. Background
   rows have no `frp`, no detection semantics, and — by construction — everyday
   rather than fire-day weather. Mixing them in would flatten exactly the
   distributions the exploration characterizes (severity, seasonality, CLC
   profiles). Nothing in 02 needs negatives.

2. **The whole 03 unsupervised block — deliberately.** The 03e verdict (k12 accepted)
   and all 03 thresholds were validated on detections only. Refitting clusters or
   anomaly detectors with ~46k everyday-weather rows would change the population —
   background days would likely dominate and form their own "no-fire" regimes,
   invalidating the acceptance gates and the elected typology. Hence the contract:
   background is **scored through the fitted objects, never included in the fit**.
   Same for 03d: DBSCAN events are structures *of detections*; a background row has
   no event semantics. The unsupervised block stays frozen; 04 consumes its outputs.

3. **The shared base CSV.** Never append background to
   `firms_spain_final_lags_index.csv` or any file the 03 notebooks read. Keep
   `background_days.csv` separate with its own key (`background_id`), joined only at
   `assemble_modeling_frame` time — otherwise the positional-join fragility flagged
   in the 2026-10-16 audit returns, and every 03 result becomes silently dependent
   on merge ordering.

4. **`versioning/catalog.yaml`.** Tracks raw external sources only (FIRMS, CLCPlus —
   verified). Background is derived data; the pipeline-chain documentation covers it.

5. **`data/raw/` and the weather cache.** Raw is never modified; background weather
   goes through the normal batch machinery into its own file under
   `data/processed/`.

## Hard requirements on the finished background file (04 depends on these)

- **Same everyday schema as positives**: the 45 columns of `EVERYDAY_FEATURES`, same
  order, raw units (no scaler reuse).
- **Own key column** (`background_id`) so all joins are by key, never row order.
- **Leakage screen documented**: count/drop of background rows within the exclusion
  distance of a same-day detection.
- **Season column derivable** from `acq_date` (`add_season`) for the season-blocked
  CV.
