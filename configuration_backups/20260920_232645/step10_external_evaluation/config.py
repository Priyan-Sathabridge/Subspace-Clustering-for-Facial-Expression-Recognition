"""Step 10 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
GROUND_TRUTH_CSV = root.GROUND_TRUTH_CSV
CLUSTER_LABELS_CSV = root.labels_csv(METHOD)
IMAGE_COLUMN = root.IMAGE_COLUMN
GROUND_TRUTH_COLUMN = root.GROUND_TRUTH_COLUMN
CLUSTER_COLUMN = root.CLUSTER_COLUMN
OUTPUT_DIR = root.step10_method_dir(METHOD)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
