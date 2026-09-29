"""Step 5 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

METHOD = root.active_method()
INPUT_CSV = root.coefficient_csv(METHOD)
OUTPUT_FOLDER = root.step5_method_dir(METHOD)
AFFINITY_METHOD = root.AFFINITY_METHOD
NORMALIZE_AFFINITY = root.NORMALIZE_AFFINITY
