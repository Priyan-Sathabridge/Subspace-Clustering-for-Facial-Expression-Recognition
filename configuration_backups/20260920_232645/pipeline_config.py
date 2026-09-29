"""Root configuration for the facial-expression subspace-clustering pipeline.

This file is the single source of truth for paths, final model parameters and
shared experimental settings.  Individual step-level config.py files should be
thin adapters that import values from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Project / data roots
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(
    os.environ.get(
        "SC_DATA_ROOT",
        "/Users/priyansathabridge/Desktop/Labelled_Expression_Data",
    )
).expanduser()

IMAGE_FOLDER = Path(
    os.environ.get("SC_IMAGE_FOLDER", str(DATA_ROOT / "all_images"))
).expanduser()

GROUND_TRUTH_CSV = Path(
    os.environ.get("SC_GROUND_TRUTH_CSV", str(DATA_ROOT / "ground_truth.csv"))
).expanduser()

# Logical stage output roots.
STEP1_ROOT = DATA_ROOT / "step_1"
STEP2_ROOT = DATA_ROOT / "step_2"
STEP3_ROOT = DATA_ROOT / "step_3"
STEP4_ROOT = DATA_ROOT / "step_4"
STEP5_ROOT = DATA_ROOT / "step_5"
STEP6_ROOT = DATA_ROOT / "step_6"
STEP7_ROOT = DATA_ROOT / "step_7"
STEP8_ROOT = DATA_ROOT / "step_8"
STEP9_ROOT = DATA_ROOT / "step_9"
STEP10_ROOT = DATA_ROOT / "step_10"
PIPELINE_LOG_ROOT = DATA_ROOT / "pipeline_logs"


# ---------------------------------------------------------------------------
# Canonical shared files
# ---------------------------------------------------------------------------

STEP1_GEOMETRY_FEATURES_CSV = STEP1_ROOT / "geometry_features.csv"
LEGACY_GEOMETRY_FEATURES_CSV = DATA_ROOT / "geometry_features.csv"
STEP2_PROCESSED_FEATURES_CSV = STEP2_ROOT / "processed_features.csv"
PCA_FEATURES_CSV = STEP3_ROOT / "pca_features.csv"


def geometry_features_csv() -> Path:
    """Return the canonical Step-1 file, with legacy fallback for resuming old runs."""
    override = os.environ.get("SC_GEOMETRY_FEATURES_CSV")
    if override:
        return Path(override).expanduser()
    if STEP1_GEOMETRY_FEATURES_CSV.exists():
        return STEP1_GEOMETRY_FEATURES_CSV
    if LEGACY_GEOMETRY_FEATURES_CSV.exists():
        return LEGACY_GEOMETRY_FEATURES_CSV
    return STEP1_GEOMETRY_FEATURES_CSV


# ---------------------------------------------------------------------------
# Methods / branch selection
# ---------------------------------------------------------------------------

ROBUST_SSC = "robust_ssc"
ROBUST_LRSC = "robust_lrsc"
FINAL_METHODS = (ROBUST_SSC, ROBUST_LRSC)
METHOD_ENV = "SC_PIPELINE_METHOD"


def normalize_method(value: str) -> str:
    value = value.strip().lower()
    aliases = {
        "ssc": ROBUST_SSC,
        "sparse": ROBUST_SSC,
        "robust_sparse": ROBUST_SSC,
        "robust_ssc": ROBUST_SSC,
        "lrsc": ROBUST_LRSC,
        "low_rank": ROBUST_LRSC,
        "robust_low_rank": ROBUST_LRSC,
        "robust_lrsc": ROBUST_LRSC,
        "both": "both",
        "all": "both",
    }
    if value not in aliases:
        raise ValueError(
            f"Unknown method '{value}'. Expected robust_ssc, robust_lrsc, or both."
        )
    return aliases[value]


def requested_methods(default: str = "both") -> list[str]:
    raw = normalize_method(os.environ.get(METHOD_ENV, default))
    return list(FINAL_METHODS) if raw == "both" else [raw]


def active_method(default: str = ROBUST_SSC) -> str:
    raw = normalize_method(os.environ.get(METHOD_ENV, default))
    if raw == "both":
        raise ValueError(
            "This stage must run once per method. Set SC_PIPELINE_METHOD to "
            "robust_ssc or robust_lrsc; the root pipeline runner does this automatically."
        )
    return raw


def method_dir(stage_root: Path, method: str) -> Path:
    method = normalize_method(method)
    if method == "both":
        raise ValueError("method_dir requires one method, not 'both'.")
    return stage_root / method


def step4_method_dir(method: str) -> Path:
    return method_dir(STEP4_ROOT, method)


def step5_method_dir(method: str) -> Path:
    return method_dir(STEP5_ROOT, method)


def step6_method_dir(method: str) -> Path:
    return method_dir(STEP6_ROOT, method)


def step7_method_dir(method: str) -> Path:
    return method_dir(STEP7_ROOT, method)


def step8_method_dir(method: str) -> Path:
    return method_dir(STEP8_ROOT, method)


def step9_method_dir(method: str) -> Path:
    return method_dir(STEP9_ROOT, method)


def step10_method_dir(method: str) -> Path:
    return method_dir(STEP10_ROOT, method)


# ---------------------------------------------------------------------------
# Step 1: feature extraction
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ENABLED_FEATURES = ("eyes", "mouth", "eyebrows")


# ---------------------------------------------------------------------------
# Step 2: processing
# ---------------------------------------------------------------------------

SAVE_VALIDATION_REPORT = True
MISSING_VALUE_METHOD = "drop"
NORMALIZATION_METHOD = "standard"
WEIGHTING_METHOD = "sqrt"


# ---------------------------------------------------------------------------
# Step 3: PCA
# ---------------------------------------------------------------------------

PCA_VARIANCE_THRESHOLD = 0.90
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Step 4: final Robust SSC / Robust LRSC
# ---------------------------------------------------------------------------

ROBUST_SSC_LAMBDA = 0.025897373396156047
ROBUST_SSC_GAMMA = 0.3986470631277377

ROBUST_LRSC_LAMBDA = 1.2247448713915892
ROBUST_LRSC_GAMMA = 0.38729833462074176

ADMM_MU = 1.0
ADMM_RHO = 1.5
ADMM_MAX_MU = 1e6
ADMM_MAX_ITER = 1000
ADMM_TOL = 1e-6
ZERO_DIAGONAL = True
L2_NORMALIZE_SAMPLES = True
COEFFICIENT_REL_TOL = 1e-6
VERBOSE = False


# ---------------------------------------------------------------------------
# Steps 5-10: affinity, clustering, visualisation and evaluation
# ---------------------------------------------------------------------------

AFFINITY_METHOD = "symmetric"
NORMALIZE_AFFINITY = True

N_CLUSTERS = 4
CLUSTERING_METHOD = "spectral"
ASSIGN_LABELS = "kmeans"

IMAGES_PER_ROW = 5
IMAGE_SIZE = (200, 200)
SHOW_FILENAMES = True
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42
HEATMAP_CMAP = "viridis"
SHOW_CLUSTER_BOUNDARIES = True

# Final-model evaluation uses the known four-expression solution.
EVALUATION_MIN_K = N_CLUSTERS
EVALUATION_MAX_K = N_CLUSTERS
SILHOUETTE_METRIC = "euclidean"

IMAGE_COLUMN = "Image"
GROUND_TRUTH_COLUMN = "Label"
CLUSTER_COLUMN = "Cluster"

# ANOVA on processed geometric features is more directly interpretable than PCs.
ANOVA_FEATURES_CSV = STEP2_PROCESSED_FEATURES_CSV


# ---------------------------------------------------------------------------
# Convenience paths
# ---------------------------------------------------------------------------


def coefficient_csv(method: str) -> Path:
    return step4_method_dir(method) / "coefficients.csv"


def error_matrix_csv(method: str) -> Path:
    return step4_method_dir(method) / "error_matrix.csv"


def affinity_csv(method: str) -> Path:
    return step5_method_dir(method) / "affinity.csv"


def clustering_dir(method: str, k: int | None = None) -> Path:
    k = N_CLUSTERS if k is None else int(k)
    return step6_method_dir(method) / f"k_{k}"


def labels_csv(method: str, k: int | None = None) -> Path:
    return clustering_dir(method, k) / "labels.csv"


def visualization_dir(method: str, k: int | None = None) -> Path:
    k = N_CLUSTERS if k is None else int(k)
    return step7_method_dir(method) / f"k_{k}"


def evaluation_results_csv(method: str) -> Path:
    return step8_method_dir(method) / "evaluation_results.csv"


def anova_csv(method: str) -> Path:
    return step9_method_dir(method) / "anova_results.csv"


def external_metrics_csv(method: str) -> Path:
    return step10_method_dir(method) / "external_metrics.csv"
