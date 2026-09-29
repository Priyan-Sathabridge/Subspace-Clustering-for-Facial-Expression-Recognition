"""
symmetric.py

Symmetric affinity matrix construction.

W = |C| + |C|^T
"""

import numpy as np

from .base import BaseAffinityMethod


class SymmetricAffinity(BaseAffinityMethod):
    """
    Construct a symmetric affinity matrix.

    Parameters
    ----------
    normalize : bool, default=True
        Scale the affinity matrix so that its
        maximum value is equal to one.
    """

    def __init__(self, normalize=True):

        self.normalize = normalize

    # -----------------------------------------------------

    def construct_affinity(self, C):
        """
        Construct a symmetric affinity matrix.

        Parameters
        ----------
        C : ndarray
            Coefficient matrix.

        Returns
        -------
        ndarray
            Symmetric affinity matrix.
        """

        C = np.asarray(C)

        if C.ndim != 2:
            raise ValueError(
                "Coefficient matrix must be two-dimensional."
            )

        if C.shape[0] != C.shape[1]:
            raise ValueError(
                "Coefficient matrix must be square."
            )

        # Absolute coefficients

        W = np.abs(C)

        # Symmetrize

        W = W + W.T

        # Remove self-connections

        np.fill_diagonal(W, 0)

        # Optional normalization

        if self.normalize:

            maximum = np.max(W)

            if maximum > 0:

                W = W / maximum

        return W