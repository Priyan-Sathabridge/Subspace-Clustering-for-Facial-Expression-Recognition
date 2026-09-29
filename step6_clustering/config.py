"""Step 6 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
INPUT_CSV = root.affinity_csv(METHOD)
PCA_FEATURES_CSV = root.PCA_FEATURES_CSV
OUTPUT_FOLDER = root.clustering_dir(METHOD, root.N_CLUSTERS)

CLUSTERING_METHOD = root.CLUSTERING_METHOD
N_CLUSTERS = root.N_CLUSTERS
ASSIGN_LABELS = root.ASSIGN_LABELS
RANDOM_STATE = root.RANDOM_STATE

# Kept for compatibility with the older iterator module.
ITERATOR_MIN_K = N_CLUSTERS
ITERATOR_MAX_K = N_CLUSTERS
