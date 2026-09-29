"""
pipeline.py

Evaluation Pipeline
"""

import numpy as np
import pandas as pd

import config

from silhouette import calculate_silhouette
from dbi import calculate_dbi
from chi import calculate_chi

from affinity_silhouette import calculate_affinity_silhouette
from affinity_ratio import calculate_affinity_ratio
from normalized_cut import calculate_normalized_cut

from eigengap import evaluate_eigengap

from results import (
    combine_results,
    save_results,
    plot_all_results,
)


# ==========================================================
# Load Feature Matrix
# ==========================================================

def load_features():

    df = pd.read_csv(
        config.FEATURES_CSV
    )

    if config.IMAGE_COLUMN not in df.columns:

        raise ValueError(
            f"'{config.IMAGE_COLUMN}' not found."
        )

    feature_columns = [

        c

        for c in df.columns

        if c != config.IMAGE_COLUMN

    ]

    X = df[
        feature_columns

    ].to_numpy(dtype=float)

    image_names = df[
        config.IMAGE_COLUMN
    ]

    return image_names, X


# ==========================================================
# Load Labels
# ==========================================================

def load_labels(label_path, image_names):
    df = pd.read_csv(label_path)
    required = {config.IMAGE_COLUMN, config.CLUSTER_COLUMN}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError('Labels require non-null image identifiers and clusters')
    if df[config.IMAGE_COLUMN].duplicated().any() or image_names.duplicated().any():
        raise ValueError('Duplicate image identifiers in features or labels')
    if set(df[config.IMAGE_COLUMN]) != set(image_names):
        raise ValueError('Features and labels must contain exactly the same images')
    # Restore PCA order before comparing labels to feature and graph rows.
    return df.set_index(config.IMAGE_COLUMN).loc[image_names, config.CLUSTER_COLUMN].to_numpy()


# ==========================================================
# Load Affinity Matrix
# ==========================================================

def load_affinity():

    W = pd.read_csv(

        config.AFFINITY_CSV

    ).to_numpy(dtype=float)

    if W.shape[0] != W.shape[1]:

        raise ValueError(
            "Affinity matrix must be square."
        )

    return W


# ==========================================================
# Main Evaluation
# ==========================================================

def run():

    print("=" * 60)
    print("Step 8 - Cluster Evaluation")
    print("=" * 60)

    # ------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------

    image_names, X = load_features()

    W = load_affinity()
    if W.shape != (len(X), len(X)) or not np.isfinite(W).all():
        raise ValueError("Affinity must be finite and match the feature sample count")

    eigengaps, eigenvalues = evaluate_eigengap(

        W,

        min_k=config.MIN_K,

        max_k=config.MAX_K

    )

    # ------------------------------------------------------

    silhouette_results = {}

    affinity_silhouette_results = {}

    affinity_ratio_results = {}

    normalized_cut_results = {}

    dbi_results = {}

    chi_results = {}

    eigengap_results = {}

    # ------------------------------------------------------
    # Iterate through clustering results
    # ------------------------------------------------------

    for k in range(

        config.MIN_K,

        config.MAX_K + 1

    ):

        print()

        print("-" * 50)

        print(f"Evaluating k = {k}")

        print("-" * 50)

        labels_csv = (

            config.CLUSTERING_FOLDER /

            f"k_{k}" /

            "labels.csv"

        )

        if not labels_csv.exists():

            raise FileNotFoundError(f"Missing labels for k={k}: {labels_csv}")

        labels = load_labels(

            labels_csv, image_names

        )

        if len(labels) != len(X):

            raise ValueError(

                "Feature matrix and labels "

                "contain different numbers "

                "of samples."

            )

        # ------------------------------------------

        silhouette_results[k] = (

            calculate_silhouette(

                X,

                labels,

                metric=config.SILHOUETTE_METRIC

            )

        )

        dbi_results[k] = (

            calculate_dbi(

                X,

                labels

            )

        )

        chi_results[k] = (

            calculate_chi(

                X,

                labels

            )

        )

        affinity_silhouette_results[k] = (

            calculate_affinity_silhouette(

                W,

                labels

            )

        )

        affinity_ratio_results[k] = (

            calculate_affinity_ratio(

                W,

                labels

            )

        )

        normalized_cut_results[k] = (

            calculate_normalized_cut(

                W,

                labels

            )

        )

        eigengap_results[k] = (

            eigengaps.get(

                k,

                np.nan

            )

        )

        print(

            f"Silhouette : "

            f"{silhouette_results[k]:.4f}"

        )

        print(

            f"DBI : "

            f"{dbi_results[k]:.4f}"

        )

        print(

            f"CHI : "

            f"{chi_results[k]:.4f}"

        )

    # ------------------------------------------------------
    # Combine
    # ------------------------------------------------------

    metric_results = {

        "Silhouette": silhouette_results,

        "Affinity Silhouette":
            affinity_silhouette_results,

        "Affinity Ratio":
            affinity_ratio_results,

        "Normalized Cut":
            normalized_cut_results,

        "DBI": dbi_results,

        "CHI": chi_results,

        "Eigengap": eigengap_results,

    }

    results_df = combine_results(

        metric_results

    )

    save_results(

        results_df,

        config.RESULTS_CSV

    )

    plot_all_results(

        results_df,

        config.PLOTS_FOLDER

    )

    print()

    print("=" * 60)

    print("Evaluation Complete")

    print("=" * 60)

    return results_df


# ==========================================================

if __name__ == "__main__":

    run()