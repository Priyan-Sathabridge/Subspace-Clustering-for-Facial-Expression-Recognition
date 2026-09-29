"""Step 2 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

INPUT_CSV = root.geometry_features_csv()
OUTPUT_FOLDER = root.STEP2_ROOT
REPORT_FOLDER = OUTPUT_FOLDER
SAVE_VALIDATION_REPORT = root.SAVE_VALIDATION_REPORT
MISSING_VALUE_METHOD = root.MISSING_VALUE_METHOD
NORMALIZATION_METHOD = root.NORMALIZATION_METHOD
WEIGHTING_METHOD = root.WEIGHTING_METHOD
