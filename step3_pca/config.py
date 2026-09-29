"""Step 3 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

INPUT_CSV = root.STEP2_PROCESSED_FEATURES_CSV
OUTPUT_FOLDER = root.STEP3_ROOT
VARIANCE_THRESHOLD = root.PCA_VARIANCE_THRESHOLD
RANDOM_STATE = root.RANDOM_STATE
