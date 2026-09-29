"""
Low Rank Representation (LRR)

Constructs a low-rank self-representation
coefficient matrix.

The coefficient matrix is later used by the
spectral clustering stage.
"""

import numpy as np

from numpy.linalg import svd

from sklearn.preprocessing import normalize

from .base import BaseCoefficientMethod


class LowRankSubspaceClustering(BaseCoefficientMethod):
    """
    Low Rank Representation (LRR).

    Parameters
    ----------
    lam : float
        Low-rank regularization parameter.

    max_iter : int
        Maximum optimization iterations.

    tol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        lam=0.1,
        max_iter=100,
        tol=1e-5
    ):

        if lam <= 0:
            raise ValueError("lam must be positive.")

        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    # ======================================================

    @staticmethod
    def singular_value_thresholding(M, tau):
        """
        Singular Value Thresholding (SVT).
        """

        U, S, VT = svd(
            M,
            full_matrices=False
        )

        S = np.maximum(S - tau, 0)

        return U @ np.diag(S) @ VT

    # ======================================================

    def compute_coefficients(self, X):
        """
        Compute the low-rank coefficient matrix.

        Parameters
        ----------
        X : ndarray
            Feature matrix
            (n_samples × n_features)

        Returns
        -------
        ndarray
            Low-rank coefficient matrix
            (n_samples × n_samples)
        """

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        X = normalize(X)

        n_samples = X.shape[0]

        identity = np.eye(n_samples)

        XtX = X @ X.T

        C = np.zeros((n_samples, n_samples))

        for _ in range(self.max_iter):

            previous = C.copy()

            C = np.linalg.solve(
                XtX + self.lam * identity,
                XtX
            )

            np.fill_diagonal(C, 0)

            C = self.singular_value_thresholding(
                C,
                self.lam
            )

            difference = np.linalg.norm(
                C - previous,
                ord="fro"
            )

            if difference < self.tol:
                break

        return C