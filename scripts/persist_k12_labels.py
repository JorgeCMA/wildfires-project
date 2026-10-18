"""Persiste la tipología k12 elegida (03e) en disco.

Reajusta la política de 03e (``notebooks/03e_validacion.ipynb`` celdas 3/17)
sobre el espacio PCA de la sonda A: StandardScaler(PC1–7) + KMeans k=12
(etiqueta dura ``cluster_meteo``) + GMM de covarianza completa k=12
(posteriores ``gmm_k12_full_p{j}``), semilla 42. Ajusta SOLO sobre los
positivos (``features_a.csv``, 23.142 detecciones): las filas de fondo se
puntuarán después con los mismos objetos ajustados vía
``K12Typology.score_background`` (contrato §B.2.2 del plan supervisado).

Uso::

    python scripts/persist_k12_labels.py
"""

from __future__ import annotations

from wildfire.data.io import load_csv, save_csv
from wildfire.processing.typology import fit_k12_typology

FEATURES_CSV = "data/processed/unsupervised/features_a.csv"
OUTPUT_CSV = "data/processed/unsupervised/cluster_labels_k12.csv"


def main() -> None:
    features = load_csv(FEATURES_CSV)
    typology = fit_k12_typology(features)
    labels = typology.labels_for(features)
    path = save_csv(labels, OUTPUT_CSV)
    print(
        f"{path} <- {labels.shape} "
        f"(k={typology.kmeans.n_clusters} groups, "
        f"seed-fitted on positives only)"
    )


if __name__ == "__main__":
    main()
