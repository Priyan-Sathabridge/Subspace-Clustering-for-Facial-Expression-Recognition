import numpy as np
from sklearn.metrics import silhouette_score


def calculate_affinity_silhouette(
        affinity_matrix,
        labels):
    """
    Silhouette score computed from the
    LRSC affinity matrix.

    Parameters
    ----------
    affinity_matrix : ndarray (n,n)

    labels : ndarray

    Returns
    -------
    float
    """

    W = affinity_matrix.astype(float)

    # Normalise affinity to [0,1]
    W = W / np.max(W)

    # Convert similarity to distance
    D = 1.0 - W

    np.fill_diagonal(D, 0.0)

    return silhouette_score(
        D,
        labels,
        metric="precomputed"
    )