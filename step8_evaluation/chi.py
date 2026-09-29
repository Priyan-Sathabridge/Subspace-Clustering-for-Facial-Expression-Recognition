# chi.py

import numpy as np

from sklearn.metrics import (
    calinski_harabasz_score
)


# ---------------------------------------------------------
# Single Clustering Evaluation
# ---------------------------------------------------------

def calculate_chi(
        X,
        labels):
    """
    Calculate the Calinski-Harabasz Index
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
        Calinski-Harabasz Index.

        Higher values indicate better
        clustering.
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
            "Warning: CHI requires at least "
            "2 clusters."
        )

        return np.nan

    if n_clusters >= len(X):

        print(
            "Warning: Number of clusters must "
            "be smaller than number of samples."
        )

        return np.nan

    score = calinski_harabasz_score(
        X,
        labels
    )

    return float(score)


# ---------------------------------------------------------
# Evaluate Multiple Values of k
# ---------------------------------------------------------

def evaluate_chi(
        X,
        labels_by_k):
    """
    Calculate Calinski-Harabasz Index values
    for multiple clustering solutions.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    labels_by_k : dict
        Dictionary mapping k to cluster labels.

    Returns
    -------
    dict
        Dictionary mapping k to CHI score.
    """

    results = {}

    for k in sorted(
            labels_by_k.keys()):

        labels = labels_by_k[k]

        score = calculate_chi(
            X,
            labels
        )

        results[k] = score

        print(
            f"k = {k}: "
            f"CHI = {score:.4f}"
        )

    return results