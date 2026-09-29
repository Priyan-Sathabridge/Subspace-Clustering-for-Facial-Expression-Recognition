"""
registry.py

Registry of clustering algorithms.
"""

from .spectral_clustering import SpectralClusteringMethod


CLUSTERING_METHODS = {

    "spectral": SpectralClusteringMethod,

}