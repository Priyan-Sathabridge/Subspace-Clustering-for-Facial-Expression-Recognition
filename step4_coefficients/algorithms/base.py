"""
base.py

Base class for all coefficient matrix construction methods.
"""

from abc import ABC, abstractmethod


class BaseCoefficientMethod(ABC):
    """
    Abstract base class for coefficient
    matrix construction algorithms.
    """

    @abstractmethod
    def compute_coefficients(self, X):
        """
        Construct the coefficient matrix.

        Parameters
        ----------
        X : ndarray
            Feature matrix
            (n_samples × n_features)

        Returns
        -------
        ndarray
            Coefficient matrix
            (n_samples × n_samples)
        """
        pass

    # -----------------------------------------------------

    @property
    def name(self):
        """
        Name of the algorithm.
        """

        return self.__class__.__name__