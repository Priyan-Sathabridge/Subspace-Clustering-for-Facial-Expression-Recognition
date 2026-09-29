# silhouette.py

import numpy as np

from sklearn.metrics import (
    silhouette_score
)


# ---------------------------------------------------------
# Single Clustering Evaluation
# ---------------------------------------------------------

def calculate_silhouette(
        X,
        labels,
        metric="euclidean"):
    """
    Calculate the Silhouette Score for one
    clustering solution.

    Parameters
    ----------
    X : array-like
        Feature matrix with shape
        (n_samples, n_features).

    labels : array-like
        Cluster labels for each sample.

    metric : str
        Distance metric used by silhouette_score.

    Returns
    -------
    float
        Silhouette Score.
        Higher values indicate better clustering.
    """

    X = np.asarray(X)

    labels = np.asarray(labels)

    unique_labels = np.unique(
        labels
    )

    n_clusters = len(
        unique_labels
    )

    # Silhouette requires at least
    # 2 distinct clusters.
    if n_clusters < 2:

        print(
            "Warning: Silhouette Score requires "
            "at least 2 clusters."
        )

        return np.nan

    # Silhouette also requires the number
    # of clusters to be smaller than the
    # number of samples.
    if n_clusters >= len(X):

        print(
            "Warning: Number of clusters must be "
            "smaller than number of samples."
        )

        return np.nan

    score = silhouette_score(
        X,
        labels,
        metric=metric
    )

    return float(score)


# ---------------------------------------------------------
# Evaluate Multiple Values of k
# ---------------------------------------------------------

def evaluate_silhouette(
        X,
        labels_by_k,
        metric="euclidean"):
    """
    Calculate Silhouette Scores for multiple
    clustering solutions.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    labels_by_k : dict
        Dictionary containing cluster labels.

        Example:

        {
            4: labels_k4,
            5: labels_k5,
            6: labels_k6
        }

    metric : str
        Distance metric.

    Returns
    -------
    dict

        {
            4: score,
            5: score,
            6: score
        }
    """

    results = {}

    for k in sorted(
            labels_by_k.keys()):

        labels = labels_by_k[k]

        score = calculate_silhouette(
            X,
            labels,
            metric=metric
        )

        results[k] = score

        print(
            f"k = {k}: "
            f"Silhouette = {score:.4f}"
        )

    return results