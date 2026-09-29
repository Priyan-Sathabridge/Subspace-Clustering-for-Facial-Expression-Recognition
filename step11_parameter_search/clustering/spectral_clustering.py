"""
spectral_clustering.py

Spectral clustering using a precomputed
affinity matrix.
"""

import numpy as np

from sklearn.cluster import SpectralClustering

from .base import BaseClusteringMethod


class SpectralClusteringMethod(BaseClusteringMethod):
    """
    Spectral clustering.

    Parameters
    ----------
    n_clusters : int

    assign_labels : str
        "kmeans"
        "discretize"

    random_state : int
    """

    def __init__(
        self,
        n_clusters,
        assign_labels="kmeans",
        random_state=42
    ):

        if n_clusters < 2:
            raise ValueError(
                "n_clusters must be at least 2."
            )

        self.n_clusters = n_clusters
        self.assign_labels = assign_labels
        self.random_state = random_state

    # -----------------------------------------------------

    def cluster(self, affinity_matrix):
        """
        Perform spectral clustering.
        """

        affinity_matrix = np.asarray(affinity_matrix)

        if affinity_matrix.ndim != 2:
            raise ValueError(
                "Affinity matrix must be two-dimensional."
            )

        if affinity_matrix.shape[0] != affinity_matrix.shape[1]:
            raise ValueError(
                "Affinity matrix must be square."
            )

        model = SpectralClustering(

            n_clusters=self.n_clusters,

            affinity="precomputed",

            assign_labels=self.assign_labels,

            random_state=self.random_state

        )

        labels = model.fit_predict(
            affinity_matrix
        )

        return labels
    # -----------------------------------------------------

 