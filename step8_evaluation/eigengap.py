# eigengap.py

import numpy as np

from scipy.sparse.csgraph import (
    laplacian
)


# ---------------------------------------------------------
# Validate Affinity Matrix
# ---------------------------------------------------------

def validate_affinity_matrix(
        affinity_matrix):
    """
    Validate and prepare an affinity matrix.

    Parameters
    ----------
    affinity_matrix : array-like
        Square affinity matrix.

    Returns
    -------
    numpy.ndarray
        Validated symmetric affinity matrix.
    """

    W = np.asarray(
        affinity_matrix,
        dtype=float
    )

    # Check matrix is 2-dimensional
    if W.ndim != 2:

        raise ValueError(
            "Affinity matrix must be "
            "2-dimensional."
        )

    # Check matrix is square
    if W.shape[0] != W.shape[1]:

        raise ValueError(
            "Affinity matrix must be square. "
            f"Received shape {W.shape}."
        )

    # Replace NaN and infinite values
    W = np.nan_to_num(
        W,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Affinity values should be non-negative
    W = np.abs(W)

    # Force symmetry
    W = (
        W + W.T
    ) / 2

    # Remove self-connections
    np.fill_diagonal(
        W,
        0
    )

    return W


# ---------------------------------------------------------
# Calculate Laplacian Eigenvalues
# ---------------------------------------------------------

def calculate_eigenvalues(
        affinity_matrix,
        normalized=True):
    """
    Calculate eigenvalues of the graph Laplacian.

    Parameters
    ----------
    affinity_matrix : array-like
        Symmetric affinity matrix.

    normalized : bool
        If True, use normalized graph Laplacian.

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues.
    """

    W = validate_affinity_matrix(
        affinity_matrix
    )

    # Construct graph Laplacian
    L = laplacian(
        W,
        normed=normalized
    )

    # L is symmetric, so eigvalsh is appropriate
    eigenvalues = np.linalg.eigvalsh(
        L
    )

    # Numerical errors can sometimes create
    # extremely small negative values.
    eigenvalues[
        np.abs(eigenvalues) < 1e-12
    ] = 0

    eigenvalues = np.sort(
        eigenvalues
    )

    return eigenvalues


# ---------------------------------------------------------
# Calculate Eigengaps
# ---------------------------------------------------------

def calculate_eigengaps(
        eigenvalues,
        min_k=4,
        max_k=20):
    """
    Calculate eigengap for candidate values of k.

    Eigengap for k is:

        lambda_(k+1) - lambda_k

    Parameters
    ----------
    eigenvalues : array-like
        Sorted Laplacian eigenvalues.

    min_k : int
        Minimum number of clusters.

    max_k : int
        Maximum number of clusters.

    Returns
    -------
    dict
        Dictionary mapping k to eigengap.
    """

    eigenvalues = np.asarray(
        eigenvalues
    )

    results = {}

    max_valid_k = min(
        max_k,
        len(eigenvalues) - 1
    )

    for k in range(
            min_k,
            max_valid_k + 1):

        # Python indexing:
        #
        # lambda_k     -> index k - 1
        # lambda_(k+1) -> index k

        gap = (
            eigenvalues[k] -
            eigenvalues[k - 1]
        )

        results[k] = float(
            gap
        )

        print(
            f"k = {k}: "
            f"Eigengap = {gap:.6f}"
        )

    return results


# ---------------------------------------------------------
# Full Eigengap Evaluation
# ---------------------------------------------------------

def evaluate_eigengap(
        affinity_matrix,
        min_k=4,
        max_k=20,
        normalized=True):
    """
    Perform complete eigengap analysis.

    Parameters
    ----------
    affinity_matrix : array-like
        Affinity matrix produced by SSC or LRSC.

    min_k : int
        Minimum candidate number of clusters.

    max_k : int
        Maximum candidate number of clusters.

    normalized : bool
        Whether to use normalized Laplacian.

    Returns
    -------
    eigengaps : dict
        Eigengap for each candidate k.

    eigenvalues : numpy.ndarray
        Sorted Laplacian eigenvalues.
    """

    eigenvalues = calculate_eigenvalues(
        affinity_matrix,
        normalized=normalized
    )

    eigengaps = calculate_eigengaps(
        eigenvalues,
        min_k=min_k,
        max_k=max_k
    )

    return eigengaps, eigenvalues