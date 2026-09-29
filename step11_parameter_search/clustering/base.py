"""
base.py

Abstract base class for clustering algorithms.
"""

from abc import ABC, abstractmethod


class BaseClusteringMethod(ABC):
    """
    Base class for clustering algorithms.
    """

    @abstractmethod
    def cluster(self, affinity_matrix):
        """
        Cluster an affinity matrix.

        Parameters
        ----------
        affinity_matrix : ndarray
            Square affinity matrix.

        Returns
        -------
        tuple
            labels,
            memberships
        """
        pass

    @property
    def name(self):
        """Algorithm name."""

        return self.__class__.__name__