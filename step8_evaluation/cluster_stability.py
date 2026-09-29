# cluster_stability.py

import numpy as np

from sklearn.metrics import (
    adjusted_rand_score
)


# ---------------------------------------------------------
# Add Noise to Feature Matrix
# ---------------------------------------------------------

def perturb_features(
        X,
        noise_level=0.01,
        random_state=None):
    """
    Add small Gaussian noise to the feature matrix.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    noise_level : float
        Standard deviation of Gaussian noise.

        This assumes X has already been
        standardized or normalized.

    random_state : int or None
        Random seed.

    Returns
    -------
    numpy.ndarray
        Perturbed feature matrix.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    rng = np.random.default_rng(
        random_state
    )

    noise = rng.normal(
        loc=0.0,
        scale=noise_level,
        size=X.shape
    )

    X_perturbed = (
        X + noise
    )

    return X_perturbed


# ---------------------------------------------------------
# Compare Two Clusterings
# ---------------------------------------------------------

def calculate_ari(
        labels_1,
        labels_2):
    """
    Calculate Adjusted Rand Index between
    two clustering solutions.

    Parameters
    ----------
    labels_1 : array-like
        First set of cluster labels.

    labels_2 : array-like
        Second set of cluster labels.

    Returns
    -------
    float
        Adjusted Rand Index.
    """

    labels_1 = np.asarray(
        labels_1
    )

    labels_2 = np.asarray(
        labels_2
    )

    if len(labels_1) != len(labels_2):

        raise ValueError(
            "Label arrays must contain "
            "the same number of samples."
        )

    return float(
        adjusted_rand_score(
            labels_1,
            labels_2
        )
    )


# ---------------------------------------------------------
# Stability for One Value of k
# ---------------------------------------------------------

def calculate_stability(
        X,
        original_labels,
        clustering_function,
        k,
        n_runs=20,
        noise_level=0.01,
        random_state=42):
    """
    Evaluate clustering stability for one
    candidate value of k.

    The feature matrix is repeatedly perturbed
    with small Gaussian noise and reclustered.

    Each new clustering is compared with the
    original clustering using ARI.

    Parameters
    ----------
    X : array-like
        Standardized or PCA-reduced feature matrix.

    original_labels : array-like
        Original cluster labels.

    clustering_function : callable
        Function used to rerun clustering.

        Expected interface:

            labels = clustering_function(
                X,
                k
            )

    k : int
        Number of clusters.

    n_runs : int
        Number of perturbation runs.

    noise_level : float
        Gaussian noise standard deviation.

    random_state : int
        Base random seed.

    Returns
    -------
    float
        Mean ARI across perturbation runs.
    """

    X = np.asarray(
        X,
        dtype=float
    )

    original_labels = np.asarray(
        original_labels
    )

    ari_scores = []

    for run in range(
            n_runs):

        seed = (
            random_state +
            run
        )

        # Perturb feature matrix
        X_perturbed = perturb_features(
            X,
            noise_level=noise_level,
            random_state=seed
        )

        # Rerun clustering
        perturbed_labels = (
            clustering_function(
                X_perturbed,
                k
            )
        )

        # Compare with original labels
        ari = calculate_ari(
            original_labels,
            perturbed_labels
        )

        ari_scores.append(
            ari
        )

    mean_ari = np.mean(
        ari_scores
    )

    return float(
        mean_ari
    )


# ---------------------------------------------------------
# Evaluate Stability Across Multiple k
# ---------------------------------------------------------

def evaluate_stability(
        X,
        labels_by_k,
        clustering_function,
        n_runs=20,
        noise_level=0.01,
        random_state=42):
    """
    Evaluate clustering stability for multiple
    candidate values of k.

    Parameters
    ----------
    X : array-like
        Feature matrix.

    labels_by_k : dict
        Dictionary mapping k to original
        clustering labels.

    clustering_function : callable
        Function that accepts:

            clustering_function(X, k)

        and returns cluster labels.

    n_runs : int
        Number of perturbation runs.

    noise_level : float
        Gaussian noise standard deviation.

    random_state : int
        Base random seed.

    Returns
    -------
    dict
        Dictionary mapping k to mean ARI.
    """

    results = {}

    for k in sorted(
            labels_by_k.keys()):

        print(
            f"\nEvaluating stability "
            f"for k = {k}"
        )

        original_labels = (
            labels_by_k[k]
        )

        score = calculate_stability(
            X=X,
            original_labels=original_labels,
            clustering_function=clustering_function,
            k=k,
            n_runs=n_runs,
            noise_level=noise_level,
            random_state=random_state
        )

        results[k] = score

        print(
            f"k = {k}: "
            f"Mean Stability ARI = "
            f"{score:.4f}"
        )

    return results