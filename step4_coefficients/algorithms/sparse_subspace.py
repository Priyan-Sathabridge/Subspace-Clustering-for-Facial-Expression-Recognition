"""
Sparse Subspace Clustering (SSC)

Constructs a sparse self-representation coefficient matrix.

The coefficient matrix is later used by the spectral
clustering stage.
"""

import numpy as np

from sklearn.linear_model import Lasso
from sklearn.preprocessing import normalize

from .base import BaseCoefficientMethod


class SparseSubspaceClustering(BaseCoefficientMethod):
    """
    Sparse Subspace Clustering (SSC).

    Parameters
    ----------
    alpha : float
        L1 regularization parameter.

    max_iter : int
        Maximum number of iterations for Lasso.
    """

    def __init__(
        self,
        alpha=0.001,
        max_iter=5000
    ):

        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        self.alpha = alpha
        self.max_iter = max_iter

    # ======================================================

    def compute_coefficients(self, X):
        """
        Compute the sparse coefficient matrix.

        Parameters
        ----------
        X : ndarray
            Feature matrix
            (n_samples × n_features)

        Returns
        -------
        ndarray
            Sparse coefficient matrix
            (n_samples × n_samples)
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        X = normalize(X)

        n_samples = X.shape[0]

        C = np.zeros((n_samples, n_samples))

        for i in range(n_samples):

            # Remove current sample

            Xi = np.delete(X, i, axis=0)

            yi = X[i]

            lasso = Lasso(
                alpha=self.alpha,
                fit_intercept=False,
                max_iter=self.max_iter
            )

            lasso.fit(Xi.T, yi)

            coef = lasso.coef_

            coef = np.insert(coef, i, 0)

            C[:, i] = coef

        return C