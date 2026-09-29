"""
base.py

Abstract base class for affinity matrix construction methods.
"""

from abc import ABC, abstractmethod


class BaseAffinityMethod(ABC):
    """
    Abstract base class for affinity matrix construction.

    Every affinity construction method must implement
    construct_affinity().
    """

    @abstractmethod
    def construct_affinity(self, C):
        """
        Construct an affinity matrix.

        Parameters
        ----------
        C : ndarray
            Coefficient matrix
            (n_samples × n_samples)

        Returns
        -------
        ndarray
            Affinity matrix
            (n_samples × n_samples)
        """
        pass

    # -----------------------------------------------------

    @property
    def name(self):
        """Return the algorithm name."""

        return self.__class__.__name__