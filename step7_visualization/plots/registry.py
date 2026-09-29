"""
registry.py

Registry of available visualization methods.
"""

from .cluster_gallery import ClusterGallery
from .umap_projection import UMAPProjection
from .affinity_heatmap import AffinityHeatmap


PLOTS = {

    "cluster_gallery": ClusterGallery,

    "umap_projection": UMAPProjection,

    "affinity_heatmap": AffinityHeatmap

}