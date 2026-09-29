# dbi.py

import numpy as np

from sklearn.metrics import (
    davies_bouldin_score
)


# ---------------------------------------------------------
# Single Clustering Evaluation
# ---------------------------------------------------------

def calculate_dbi(
        X,
        labels):
    """
    Calculate the Davies-Bouldin Index
    for one clustering solution.

    Parameters
    ----------
    X : array-like
        Feature matrix with shape
        (n_samples, n_features).

    labels : array-like
        Cluster labels.

    Returns
    -------
    float
        Davies-Bouldin Index.

        Lower values indicate better
        cluster separation.
    """

    X = np.asarray(X)

    labels = np.asarray(labels)

    unique_labels = np.unique(
        labels
    )

    n_clusters = len(
        unique_labels
    )

    if n_clusters < 2:

        print(
            "Warning: DBI requires at least "
            "2 clusters."
        )

        return np.nan

    if n_clusters >= len(X):

        print(
            "Warning: Number of clusters must "
            "be smaller than number of samples."
        )

        return np.nan

    score = davies_bouldin_score(
        X,
        labels
    )

    return float(score)


# ---------------------------------------------------------
# Evaluate Multiple Values of k
# ---------------------------------------------------------

def evaluate_dbi(
        X,
        labels_by_k):
    """
    Calculate Davies-Bouldin Index values
    for multiple clustering solutions.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    labels_by_k : dict
        Dictionary mapping k to labels.

    Returns
    -------
    dict
        Dictionary mapping k to DBI score.
    """

    results = {}

    for k in sorted(
            labels_by_k.keys()):

        labels = labels_by_k[k]

        score = calculate_dbi(
            X,
            labels
        )

        results[k] = score

        print(
            f"k = {k}: "
            f"DBI = {score:.4f}"
        )

    return results