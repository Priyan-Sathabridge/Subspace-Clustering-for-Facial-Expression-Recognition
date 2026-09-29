"""Step 9 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
FEATURE_CSV = root.ANOVA_FEATURES_CSV
CLUSTER_CSV = root.labels_csv(METHOD)
OUTPUT_CSV = root.anova_csv(METHOD)
IMAGE_COLUMN = root.IMAGE_COLUMN
CLUSTER_COLUMN = root.CLUSTER_COLUMN
