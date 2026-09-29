"""
base.py

Abstract base class for visualization methods.
"""

from abc import ABC, abstractmethod


class BasePlot(ABC):
    """
    Abstract base class for all visualization methods.

    Every plot should implement the create() method.
    """

    @abstractmethod
    def create(self):
        """
        Generate and save the visualization.
        """
        pass

    @property
    def name(self):
        """
        Return the plot name.
        """
        return self.__class__.__name__