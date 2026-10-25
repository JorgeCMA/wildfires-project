# Deployment unit: relative risk bands (A) + empirical CCAA-day rate (B)

Companion to `docs/supervised-training-temporal-ratio-experiment.md`. The
occurrence model (block 04) outputs a **ranking, not a calibrated probability**
(prevalence is set by the negative sampling, `background_ratio`, not observed).
This note documents the first two deployment-facing additions built on top of
that ranking. All pipeline code lives in `notebooks/04_aprendizaje_supervisado.ipynb`;
the tested units live in `src/wildfire/analysis/deployment.py`.

- **A** — `risk_band` / `risk_percentile`: a *relative* categorical risk level.
- **B** — `ccaa_day_fire_rates.csv` + `ccaa_fire_rate`: an *absolute, measured*
  regional base rate.

They are complementary: A tells you which rows stand out **within the scored
batch**, B tells you how fire-prone the **region** is **at all**.

---

## A. Relative risk bands

`add_risk_bands(frame, *, score_col="score", group_col="protocol", n_bands=5, labels=None)`
appends two columns to the out-of-fold prediction frame:

| Column | Meaning |
|---|---|
| `risk_band` | Ordered category, default `Muy bajo / Bajo / Medio / Alto / Muy alto` |
| `risk_percentile` | Rank-based percentile (0–100) of the score |

Design choices:

- **Five bands, Spanish labels** (`RISK_BAND_LABELS[5]` in
  `src/wildfire/analysis/deployment.py`), set by `modeling.occurrence.risk_bands`
  in `configs/project.yaml` and validated in `occurrence_settings()`.
- **Band per `protocol`** (blocked validation protocol), because scores from
  different folds are not on a common scale. With `group_col=None` the whole
  frame is banded together.
- **Rank-based** (`quantile_bands` uses `Series.rank(pct=True)`), so identical
  scores always share a band — ties never split across bands.
- Quintiles are equal-sized by construction (94,074 rows → ~9,408 per band per
  protocol).

### The caveat that must travel with the column

`risk_band` is **relative to the batch**. "Muy alto" means the row is high
compared with the other rows scored together in that protocol — **not** an
absolute danger level. Without a calibrated probability (see the C track,
planned) the band cannot be read as "this place is in high danger" out of
context.

---

## B. Empirical CCAA-day fire rate

`ccaa_day_fire_rates(positives)` computes `P(≥1 detection | CCAA, day)` from the
FIRMS detections against the **full** CCAA × day calendar (every day in the
observations' span, per region). It needs no extra API quota — it is arithmetic
on dates already on disk.

Now persisted by notebook 04 to `data/processed/predictions/ccaa_day_fire_rates.csv`
(config key `output.ccaa_day_fire_rates`), with columns
`ccaa, n_ccaa_days, n_fire_days, fire_rate` plus a pooled `ALL` row.

`join_ccaa_fire_rates(records, ccaa_of_record, rates)` attaches the per-region
rate to each scored row as `ccaa_fire_rate` (the pooled `ALL` row is excluded;
unknown regions become NaN). The ranking `score` is untouched.

Current table (2023–2024, 731 days per region), highest first:

| CCAA | fire_rate | CCAA | fire_rate |
|---|---|---|---|
| Andalucía | 0.668 | Navarra | 0.153 |
| Castilla-La Mancha | 0.497 | Cantabria | 0.146 |
| Extremadura | 0.456 | Canarias | 0.129 |
| Castilla y León | 0.393 | Madrid | 0.098 |
| Comunidad Valenciana | 0.384 | País Vasco | 0.092 |
| Cataluña | 0.354 | Islas Baleares | 0.063 |
| Aragón | 0.320 | La Rioja | 0.053 |
| Región de Murcia | 0.294 | **ALL (pooled)** | **0.268** |
| Asturias | 0.279 | Galicia | 0.185 |

This is an honest **base rate**: it does not respond to a specific day's weather
or a specific spot's terrain — the model's ranking does. Use them together.

---

## Fold audit: `season/2024-Q1`

The blocked-fold metrics show `season/2024-Q1` scoring almost perfectly
(PR-AUC 0.991, ROC-AUC 0.996) versus `2023-Q1` (0.958 / 0.927). Before trusting
it, the notebook (section 6) runs four checks:

| Check | Result |
|---|---|
| Banned (satellite/sub-daily) columns in `X` | none |
| Trivial control (`month` / `day_of_year` / `ccaa` only) | PR-AUC **0.451** — does **not** reproduce the score |
| Per-class feature profile | separation driven by real weather (RH, shortwave, VPD, temperature) |
| Permutation importance (that fold) | `boundary_layer_height` dominates (0.33), then `clc_class_6`, `precipitation` |

**Verdict (printed by the notebook): genuinely separable, not a calendar/CCAA
artifact.** The control model, which sees only calendar + region, cannot
separate the fold, so the near-perfect score comes from the everyday weather and
land-cover features — physically plausible.

**Watch item:** `boundary_layer_height` is the top permutation feature by a wide
margin, and it is the one variable with upstream nulls (~8,912 rows, median-
imputed). Its dominance deserves a follow-up before the number is over-trusted;
it is flagged, not (yet) explained.

---

## Output contract — `occurrence_scores.csv`

One row per `(record, protocol)`; 94,074 rows, 9 columns:

| Column | Meaning |
|---|---|
| `record_id` | `detection_id` (positive) or `background_id` (negative) |
| `label` | 1 = fire detection, 0 = background day |
| `score` | Out-of-fold ranking score (0–1); **not** a probability |
| `protocol` | Blocking used (`season` or `ccaa`) |
| `held_out` | The held-out group for that row's fold |
| `risk_band` | Relative 5-level category (A) |
| `risk_percentile` | Relative 0–100 percentile (A) |
| `ccaa` | Region of the row |
| `ccaa_fire_rate` | Empirical CCAA-day base rate (B) |

---

## Reproduce

```bash
# regenerate the artifacts end-to-end (writes occurrence_scores.csv,
# occurrence_fold_metrics.csv, ccaa_day_fire_rates.csv)
python -c "import nbformat; from nbclient import NotebookClient; \
nb=nbformat.read('notebooks/04_aprendizaje_supervisado.ipynb', as_version=4); \
NotebookClient(nb, timeout=1800, resources={'metadata':{'path':'notebooks'}}).execute(); \
nbformat.write(nb, 'notebooks/04_aprendizaje_supervisado.ipynb')"

# unit tests
pytest tests/test_deployment.py
```

---

## Limits & what is *not* here

- **No probability.** Bands are relative; `ccaa_fire_rate` is a regional base
  rate, not `P(fire at this point)`.
- **Frozen pool.** The background pool is complete and frozen (2026-10-25;
  ratio ≥ 2 per year), so these numbers are final for that snapshot.
- **Base rate is coarse.** Region × day only; it ignores the day's conditions and
  the exact location.
- **Offline only.** These are historical artifacts; the live scorer lives in the
  demo (`demo/app.py` + `analysis/model_bundle.py`).

## Backlog

- **C — calibrated exceedance probability** (next): sweep a 5 / 10 / 20 km
  margin, event-dedup (`unsupervised/eventos.csv`), isotonic calibration fit on
  2024 with the model trained on 2023, prior-corrected to the measured base rate,
  blocked CIs. See `docs/calibration-C.md` (to be written).
- **Offline demo** (static report, no new deps): CCAA base-rate choropleth, band
  distribution, PR curves, fold-audit verdict; extended with C's reliability
  curves. Tracked in `TODO.md`.
