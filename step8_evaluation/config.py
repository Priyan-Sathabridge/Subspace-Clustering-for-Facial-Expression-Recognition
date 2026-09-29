"""Step 8 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
FEATURES_CSV = root.PCA_FEATURES_CSV
AFFINITY_CSV = root.affinity_csv(METHOD)
CLUSTERING_FOLDER = root.step6_method_dir(METHOD)
OUTPUT_FOLDER = root.step8_method_dir(METHOD)
RESULTS_CSV = OUTPUT_FOLDER / "evaluation_results.csv"
PLOTS_FOLDER = OUTPUT_FOLDER / "plots"
MIN_K = root.EVALUATION_MIN_K
MAX_K = root.EVALUATION_MAX_K
IMAGE_COLUMN = root.IMAGE_COLUMN
CLUSTER_COLUMN = root.CLUSTER_COLUMN
SILHOUETTE_METRIC = root.SILHOUETTE_METRIC
