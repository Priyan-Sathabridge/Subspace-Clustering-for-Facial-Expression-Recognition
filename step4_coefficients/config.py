"""Step 4 adapter: selected optimal Robust SSC / Robust LRSC models."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pipeline_config as root

INPUT_CSV = root.PCA_FEATURES_CSV
OUTPUT_ROOT = root.STEP4_ROOT
COEFFICIENT_METHODS = root.requested_methods(default="both")

ROBUST_SSC_LAMBDA = root.ROBUST_SSC_LAMBDA
ROBUST_SSC_GAMMA = root.ROBUST_SSC_GAMMA
ROBUST_LRSC_LAMBDA = root.ROBUST_LRSC_LAMBDA
ROBUST_LRSC_GAMMA = root.ROBUST_LRSC_GAMMA

ADMM_MU = root.ADMM_MU
ADMM_RHO = root.ADMM_RHO
ADMM_MAX_MU = root.ADMM_MAX_MU
ADMM_MAX_ITER = root.ADMM_MAX_ITER
ADMM_TOL = root.ADMM_TOL
ZERO_DIAGONAL = root.ZERO_DIAGONAL
L2_NORMALIZE_SAMPLES = root.L2_NORMALIZE_SAMPLES
VERBOSE = root.VERBOSE
COEFFICIENT_REL_TOL = root.COEFFICIENT_REL_TOL
