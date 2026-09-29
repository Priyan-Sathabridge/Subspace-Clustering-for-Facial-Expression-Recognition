"""Step 7 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
IMAGE_FOLDER = root.IMAGE_FOLDER
PCA_FEATURES_CSV = root.PCA_FEATURES_CSV
AFFINITY_CSV = root.affinity_csv(METHOD)
LABELS_CSV = root.labels_csv(METHOD)
OUTPUT_FOLDER = root.visualization_dir(METHOD)

IMAGES_PER_ROW = root.IMAGES_PER_ROW
IMAGE_SIZE = root.IMAGE_SIZE
SHOW_FILENAMES = root.SHOW_FILENAMES
UMAP_NEIGHBORS = root.UMAP_NEIGHBORS
UMAP_MIN_DIST = root.UMAP_MIN_DIST
UMAP_RANDOM_STATE = root.UMAP_RANDOM_STATE
HEATMAP_CMAP = root.HEATMAP_CMAP
SHOW_CLUSTER_BOUNDARIES = root.SHOW_CLUSTER_BOUNDARIES

# Compatibility with the iterator module; the root pipeline uses run.py.
ITERATOR_MIN_K = root.N_CLUSTERS
ITERATOR_MAX_K = root.N_CLUSTERS
CLUSTERING_FOLDER = root.step6_method_dir(METHOD)
