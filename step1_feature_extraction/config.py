"""Step 1 adapter to the root pipeline configuration."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

IMAGE_FOLDER = root.IMAGE_FOLDER
OUTPUT_FOLDER = root.STEP1_ROOT
IMAGE_EXTENSIONS = root.IMAGE_EXTENSIONS
ENABLED_FEATURES = list(root.ENABLED_FEATURES)
