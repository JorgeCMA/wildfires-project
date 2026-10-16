# Next unsupervised implementations — 03d eventos, 03e validación, reintegration

**Status**: execution plan for the remaining pieces of
`plan_unsupervised_implementation.md` (the working plan — §§1–7 there are *not*
restated here; this file only sequences what is left). Decisions closed here are
marked **[DECIDED]**; open ones, **[OPEN]**.

**Base (2026-10-16)**: 23,142 detections, not 30,412 — the 01b-2.11 quemas filter
(inside-window Oct15–May30 + CLC 6/7 → 7,270 apartadas) is now decided alongside
`clc!=1` + `type==0`. Reference numbers on the new base: BLH imputation spike
3,065 rows at 577.5; FRP top-15% at 25.6. `sup_forestal_ha` is kept in the frame
(58-col `cleaned`, 61-col `lags_index`); it enters no input set.

**Preconditions (in flight / verify before starting)** — fixes to 03a/03b/03c
decided during the 2026-10-14 review:

1. `features_a.csv` carries the **7 decorrelated weather PCs** (90.9 % variance),
   not the 14 raw collinear numerics — per the working plan §5-A recommendation.
   Sets B/C inherit the PCs.
2. `cluster_labels.csv` exists with hard labels **and GMM `predict_proba` soft
   memberships** (03b markdown already promised it to 03e).
3. `anomalias_{a,b}.csv` have **unique, self-describing column names**
   (`*_score` / `*_flag`), not suffixed duplicates.
4. 03a no longer advertises set D; set D lives in 03d (below) **[DECIDED]**.
5. All of 03a → 03b → 03c re-executed in order after these edits, markdown
   claims re-synced to the new results.

---

## 1. 03d_eventos — spatio-temporal events (working plan §6.3)

**Question**: how many real fires stand behind 23,142 detections, and does one
event span several conditions-regimes?

**Input**: `data/processed/expanded/firms_spain_final_lags_index.csv` (raw
units; no scaler reuse — hard rule 4.3). **No `features_d.csv`**: the set-D
matrix is tiny and specific to this analysis, so 03d builds it inline
**[DECIDED]**:

- `latitude`, `longitude` → EPSG:3035 (LAEA Europe, metres) → km, via pyproj
  (`Transformer`, the `clc_enrichment.py` precedent — UTM 30N does not cover
  Canarias, so 25830 would distort DBSCAN distances there) **[DECIDED]**.
- `acq_date` → continuous day index (`day - day.min()`, integer days).
- Anisotropic scaling: **1 day ≈ 5–10 km**. Default 7.5 km/day **[DECIDED]**, with
  a sensitivity cell at {5, 7.5, 10} because the event count depends on it.

**Method**: DBSCAN (sklearn; installed) on `(x_km, y_km, day_scaled)`.
`min_samples = 3` (a 2-detection "event" is a re-detection, not an event
structure) **[DECIDED, provisional]**; `eps` chosen from the k-distance knee
(plot shipped as a figure). HDBSCAN remains **[OPEN]** — only if DBSCAN's single
density visibly fails (mixed regime sizes), and only after the dependency is
approved per working-plan §6.

**Outputs**:

- `data/processed/unsupervised/eventos.csv` — one row per detection with
  `event_id` (−1 = noise), kept on the common index with the features CSVs.
- `data/processed/unsupervised/eventos_resumen.csv` — **event-level table**:
  `event_id`, n detections, first/last day, duration, bbox/centroid, CCAA mix,
  max `frp`, mean of the set-A PCs (averaged meteorology per the plan).
- Summary cells: event count, size/duration distributions, noise fraction.

**Cross-reference (the interesting part)**: join `event_id` against 03b cluster
labels — does one event span several regimes? Report the distribution of
distinct clusters per event. And against 03c anomaly flags — are multi-day
events anomaly-dense or anomaly-poor?

**Gotchas**: FIRMS overpass gaps mean one fire can skip days — the day≈km
scaling is the only thing bridging that, hence the sensitivity cell. Noise
points (isolated agricultural burns) are a *result*, not a failure: report
their CLC/CCAA profile.

## 2. 03e_validacion — stability (§7A) + utility (§7B)

**Question**: does the typology exist (7A) and would it help prediction (7B)?
A typology is accepted only if it passes both.

**Inputs**: `features_{a,b,c}.csv` (PC-based), `perfilado.csv`,
`cluster_labels.csv` (needs the GMM probabilities — precondition 2),
`anomalias_{a,b}.csv` (unique columns — precondition 3), `eventos.csv`.

**§7A — does the structure exist?** (acceptance thresholds from the working plan)

| Check | Implementation | Acceptance |
|---|---|---|
| Resample stability | 30 resamples at 80 %, refit, ARI vs full-data labels, `random_state=42` base | mean ARI > 0.7 at chosen k |
| Spatial stability | leave-one-CCAA-out: refit, reassign held-out CCAA | no group disappears without a dominant region |
| Temporal replication | fit 2023 → assign 2024; and reverse | similar group profiles (per-group means in original units) |
| Preprocessing robustness | Standard vs Robust, ±BLH, ±log1p — 03a already produced the first ARI; extend to the full grid | ARI > 0.6 across variants |
| Physical sense | high VPD ↔ low humidity ↔ high ET0 per group | manual review, documented |

**§7B — would it help prediction?** (indirect until the ablation contract)

| Check | Implementation | Acceptance |
|---|---|---|
| Severity separation | Kruskal on `frp` across groups + max pairwise Cliff's \|δ\| | ≥ one pair with \|δ\| ≥ `unsupervised.min_cliff_delta` (config, default 0.2) |
| Negative controls | same severity test for month-only and CCAA-only partitions | candidate beats **both** |
| Non-coincidence | ARI of candidate vs month and vs CCAA | low on both |
| Non-triviality | normalised MI vs `ccaa`, `month`, `year`, `daynight` | reported; > 0.5 with one variable = the group *is* that variable |
| GMM regime overlap | use the saved `P(GMM)` columns: how much of the data sits below, say, 0.8 max-posterior? | reported — quantifies "regimes overlap" claim |

**Output**: `data/processed/unsupervised/validacion_resumen.csv` (one row per
check × variant, pass/fail), plus the final typology cell: chosen k, chosen
set (A/B/C), per-group profile table in original units, and an explicit
verdict — **accepted / rejected / inconclusive** — against 7A ∧ 7B.

**Gotcha**: no pure-random splits anywhere in this notebook (autocorrelation);
CCAA/year blocks only (working plan §7A note). `blocked_splits` from
`occurrence.py` is available if a grouping helper is needed.

## 3. Reintegration (only if 03e accepts)

Working plan §8.6, made concrete:

1. Add `cluster_meteo` (typology label) and `score_anomalia` (03c consensus
   score) as **candidate** columns for the occurrence model — fit on the
   features only, never with test labels.
2. Wire them through `occurrence.py`'s contract deliberately: they must be
   computable for background days (CCAA × day grid) too — clusters/anomaly
   scores are functions of everyday weather, so this holds, but the background
   rows must be *scored by the same fitted objects* (scaler + GMM/IF refit
   policy documented).
3. Final promotion is subject to the §7B **ablation contract**: ΔPR-AUC
   with/without the new columns under `blocked_splits`. Until that runs, the
   columns exist but are declared unproven in the methods note.

## 4. Deliverables checklist (working plan §8, updated)

- [x] `features_{a,b,c}.csv` + `perfilado.csv` (03a; PC-based after fixes)
- [x] `cluster_labels.csv` — id, k, **P(GMM)**, score (03b; after fixes)
- [x] `anomalias_{a,b}.csv` with unique columns (03c; after fixes)
- [x] `eventos.csv` + `eventos_resumen.csv` (03d; 1.177 eventos, 21,4 % ruido)
- [x] `validacion_resumen.csv` + veredicto: A/k12 aceptada, k5/k7 rechazadas (03e)
- [ ] Figures: BIC/silhouette (done), map per group (done), anomalies (done),
      k-distance + event map (03d), stability/utility summary (03e)
- [ ] Methods note including working-plan §2 honest limits, updated with the
      PC-based set A and the validation verdict
- [ ] 03 index notebook updated: 03b/03c no longer "planificado", correct
      `03c_anomalias` name, add 03d/03e rows as they land

## 5. Open decisions to close before/during 03d–03e

- **[OPEN]** HDBSCAN / umap / kmodes dependencies — only if sklearn DBSCAN or
  the set-C `clc_class` handling leaves a concrete unanswered question
  (working plan §6).
- **[OPEN]** `min_samples` for DBSCAN — 3 provisional; revisit if the noise
  fraction is implausibly high (> 30 %) or events look over-merged.
- **[OPEN]** `riesgo_antropico_alto` as a set-C variant — only if the typology
  passes 7A (working plan §5 note).
- **[DROPPED]** working-plan §6.4 virulence-inside variant (`log1p(frp)` as an
  input) — violates hard rule 4.1 (no satellite in inputs); the severity
  analysis in 03b/§7B answers the underlying question indirectly.
