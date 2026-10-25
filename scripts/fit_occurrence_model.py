"""Fit and persist the occurrence-model bundle for the demo's live scorer.

Reproduces the notebook-04 deployment slice end to end without the notebook
loop: train the chosen estimator on 2023, calibrate the exceedance probability
on 2024 (isotonic + prior correction against the measured background base
rate), and dump a single joblib bundle — preprocessing, estimator and the
per-margin calibrators — under ``output.model_bundle``.

The persisted estimator is the **2023-trained** one, the exact model whose
scores the calibrators were fit on (the notebook's route C uses the same
split), so a live probability stays coherent with its calibration.

    python scripts/fit_occurrence_model.py
    python scripts/fit_occurrence_model.py --model random_forest --output path.joblib

The bundle is versioned (one ~400 KB joblib) and reproducible: rerun this script
after changing the config or the training data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from wildfire.analysis.calibration import calibrate_exceedance, exceedance_summary
from wildfire.analysis.model_bundle import (
    MarginCalibrator,
    OccurrenceBundle,
    fit_preprocessing,
    save_bundle,
    transform,
)
from wildfire.analysis.occurrence_model import categorical_columns, model_candidates
from wildfire.config import load_config
from wildfire.data.io import load_csv
from wildfire.processing.occurrence import (
    EVERYDAY_FEATURES,
    LABEL_COL,
    assemble_modeling_frame,
    load_background_days,
    occurrence_features,
    occurrence_settings,
    sample_background,
    validate_background_days,
)

#: Positives file (same input notebook 04 uses).
POSITIVES_CSV = "data/processed/expanded/firms_spain_final_lags_index.csv"

#: Temporal split: train on the earlier year, calibrate on the later one.
TRAIN_YEAR, CALIB_YEAR = 2023, 2024

#: Cross-fit folds for the reliability estimate (matches notebook 04).
N_SPLITS = 5


def _years(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame["acq_date"], errors="coerce").dt.year.to_numpy()


def fit_bundle(
    model_name: str = "hist_gradient_boosting",
) -> tuple[OccurrenceBundle, pd.DataFrame]:
    """Fit the deployment bundle and return it with its per-margin summary."""
    settings = occurrence_settings()
    ratio = int(settings["background_ratio"])
    seed = int(settings["random_state"])
    leakage_max_km = float(settings.get("leakage_max_km", 1.0))
    margins = tuple(
        float(m)
        for m in load_config()["modeling"]["occurrence"]["calibration_margins_km"]
    )

    positives = load_csv(POSITIVES_CSV)
    background = load_background_days()
    validate_background_days(background, positives, leakage_max_km)

    pos_year, bg_year = _years(positives), _years(background)
    train_pos = positives.loc[pos_year == TRAIN_YEAR]
    calib_pos = positives.loc[pos_year == CALIB_YEAR]
    train_neg = sample_background(
        background.loc[bg_year == TRAIN_YEAR],
        n_positives=len(train_pos),
        ratio=ratio,
        seed=seed,
    )
    calib_neg = sample_background(
        background.loc[bg_year == CALIB_YEAR],
        n_positives=len(calib_pos),
        ratio=ratio,
        seed=seed,
    )

    train_frame = assemble_modeling_frame(
        occurrence_features(train_pos), occurrence_features(train_neg)
    )
    calib_frame = assemble_modeling_frame(
        occurrence_features(calib_pos), occurrence_features(calib_neg)
    )

    preprocessing = fit_preprocessing(
        train_frame, EVERYDAY_FEATURES, categorical_columns()
    )
    x_train = transform(train_frame, preprocessing)
    x_calib = transform(calib_frame, preprocessing)

    estimator = model_candidates(seed=seed)[model_name]
    estimator.fit(x_train, train_frame[LABEL_COL])

    # Re-attach the calibration dates (assemble_modeling_frame keeps only the
    # everyday block) so margin labeling and day-blocked cross-fitting work.
    calib_frame = calib_frame.copy()
    calib_frame["acq_date"] = np.concatenate(
        [calib_pos["acq_date"].to_numpy(), calib_neg["acq_date"].to_numpy()]
    )
    calib_frame["score"] = estimator.predict_proba(x_calib)[:, 1]

    results = calibrate_exceedance(
        calib_frame,
        positives,
        background,
        margins=margins,
        n_splits=N_SPLITS,
        seed=seed,
    )
    summary = exceedance_summary(results)

    margin_calibrators = {
        float(margin): MarginCalibrator(
            margin_km=float(margin),
            calibrator=result.calibrator,
            source_prevalence=float(result.source_prevalence),
            target_prevalence=float(result.target_prevalence),
        )
        for margin, result in results.items()
    }

    bundle = OccurrenceBundle(
        model=estimator,
        preprocessing=preprocessing,
        margins=margin_calibrators,
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_name": model_name,
            "train_year": TRAIN_YEAR,
            "calib_year": CALIB_YEAR,
            "background_ratio": ratio,
            "random_state": seed,
            "n_train": len(train_frame),
            "n_calib": len(calib_frame),
            "n_train_positives": len(train_pos),
            "n_calib_positives": len(calib_pos),
            "margins_km": sorted(margin_calibrators),
            "feature_cols": list(EVERYDAY_FEATURES),
            "calibration": summary.to_dict(orient="records"),
        },
    )
    return bundle, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="hist_gradient_boosting",
        help="estimator key from wildfire.analysis.occurrence_model.model_candidates",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="bundle path (default: output.model_bundle from configs/project.yaml)",
    )
    args = parser.parse_args()

    bundle, summary = fit_bundle(args.model)

    output = args.output or load_config()["output"]["model_bundle"]
    path = save_bundle(bundle, output)

    meta = bundle.metadata
    print(
        f"Modelo {meta['model_name']} | entreno {meta['train_year']}: "
        f"{meta['n_train']:,} filas | calibra {meta['calib_year']}: "
        f"{meta['n_calib']:,} filas"
    )
    print(summary.to_string(index=False))
    print(f"Bundle -> {path}")


if __name__ == "__main__":
    main()
