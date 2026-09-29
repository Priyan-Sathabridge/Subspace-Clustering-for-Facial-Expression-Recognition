"""Step 4: construct final optimal Robust SSC and Robust LRSC matrices.

This replaces the old LASSO-only SSC / SVT-style LRR Step 4 path for the
final comparison experiment. Both models are fitted to exactly the same PCA
feature matrix with the same sample L2 normalisation used in Step 11.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import *
from algorithms.robust_sparse import RobustSparseSubspaceClustering
from algorithms.robust_low_rank import RobustLowRankSubspaceClustering


NON_FEATURE_COLUMNS = {
    "Image", "image", "Label", "label", "Path", "path", "Cluster", "cluster"
}


def load_feature_matrix(path: Path):
    df = pd.read_csv(path)

    image_col = "Image" if "Image" in df.columns else ("image" if "image" in df.columns else None)
    if image_col is not None:
        if df[image_col].duplicated().any():
            raise ValueError("Duplicate image identifiers found in PCA feature file.")
        image_ids = df[image_col].astype(str).to_numpy()
    else:
        image_ids = np.arange(len(df)).astype(str)
        print("WARNING: no Image column found; row numbers will be used as image identifiers.")

    feature_df = df.drop(
        columns=[c for c in NON_FEATURE_COLUMNS if c in df.columns],
        errors="ignore",
    ).select_dtypes(include=[np.number])

    if feature_df.shape[1] == 0:
        raise ValueError("No numeric PCA feature columns were found.")
    if feature_df.isna().any().any():
        raise ValueError("PCA feature matrix contains missing values.")

    X = feature_df.to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("PCA feature matrix contains NaN or infinite values.")

    return X, image_ids, list(feature_df.columns)


def build_algorithm(method: str):
    method = method.lower()

    common = dict(
        mu=ADMM_MU,
        rho=ADMM_RHO,
        max_mu=ADMM_MAX_MU,
        max_iter=ADMM_MAX_ITER,
        tol=ADMM_TOL,
        zero_diagonal=ZERO_DIAGONAL,
        l2_normalize_samples=L2_NORMALIZE_SAMPLES,
        verbose=VERBOSE,
    )

    if method == "robust_ssc":
        return RobustSparseSubspaceClustering(
            lambda_reg=ROBUST_SSC_LAMBDA,
            gamma=ROBUST_SSC_GAMMA,
            **common,
        )

    if method == "robust_lrsc":
        return RobustLowRankSubspaceClustering(
            lambda_reg=ROBUST_LRSC_LAMBDA,
            gamma=ROBUST_LRSC_GAMMA,
            **common,
        )

    raise ValueError(f"Unknown final coefficient method: {method}")


def coefficient_structure(C: np.ndarray):
    max_abs = float(np.max(np.abs(C))) if C.size else 0.0
    threshold = max(COEFFICIENT_REL_TOL * max_abs, 1e-10)
    active = int(np.sum(np.abs(C) > threshold))
    total = int(C.size)
    density = active / total if total else 0.0
    return threshold, active, density, 1.0 - density


def effective_rank(C: np.ndarray):
    s = np.linalg.svd(C, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int(np.sum(s > COEFFICIENT_REL_TOL * s[0]))


def save_model_outputs(method: str, algorithm, C, X_rows, image_ids, feature_names):
    out = OUTPUT_ROOT / method
    out.mkdir(parents=True, exist_ok=True)

    # Numeric-only coefficient matrix retained for Step 5 compatibility.
    pd.DataFrame(C).to_csv(out / "coefficients.csv", index=False)

    # E is features x samples, matching the mathematical convention.
    pd.DataFrame(algorithm.error_, index=feature_names, columns=image_ids).to_csv(
        out / "error_matrix.csv"
    )

    # Keep row/image alignment explicitly available for later cluster comparison.
    pd.DataFrame({"Row": np.arange(len(image_ids)), "Image": image_ids}).to_csv(
        out / "image_order.csv", index=False
    )

    threshold, active, density, sparsity = coefficient_structure(C)
    X_math = algorithm.normalized_rows_.T
    eps = np.finfo(float).eps
    rel_error = np.linalg.norm(algorithm.error_, ord="fro") / max(
        np.linalg.norm(X_math, ord="fro"), eps
    )
    rel_reconstruction = np.linalg.norm(
        X_math - X_math @ C - algorithm.error_, ord="fro"
    ) / max(np.linalg.norm(X_math, ord="fro"), eps)

    rows = [
        ("Algorithm", algorithm.name),
        ("Method_Key", method),
        ("Input_CSV", str(INPUT_CSV)),
        ("Samples", X_rows.shape[0]),
        ("Features", X_rows.shape[1]),
        ("Coefficient_Rows", C.shape[0]),
        ("Coefficient_Columns", C.shape[1]),
        ("lambda", algorithm.lambda_reg),
        ("gamma", algorithm.gamma),
        ("L2_Normalize_Samples", algorithm.l2_normalize_samples),
        ("Zero_Diagonal", algorithm.zero_diagonal),
        ("Converged", algorithm.converged_),
        ("Iterations", algorithm.n_iter_),
        ("Final_Constraint_Error", algorithm.constraint_error_),
        ("Final_Parameter_Change", algorithm.parameter_change_),
        ("Relative_Reconstruction", rel_reconstruction),
        ("Relative_Error", rel_error),
        ("Coefficient_Threshold", threshold),
        ("Coefficient_Active", active),
        ("Coefficient_Density", density),
        ("Coefficient_Sparsity", sparsity),
        ("Effective_Rank_C", effective_rank(C)),
    ]
    pd.DataFrame(rows, columns=["Property", "Value"]).to_csv(
        out / "coefficient_info.csv", index=False
    )

    # Keep diagnostics for debugging failed fits, then stop the runner. Output
    # presence alone is therefore not evidence of a converged model.
    if not algorithm.converged_:
        raise RuntimeError(
            f"{algorithm.name} did not converge at the selected optimal parameters. "
            "Do not continue to affinity construction until this is resolved."
        )

    print(f"Saved {method} outputs to: {out}")
    print(
        f"  lambda={algorithm.lambda_reg:.8g}, gamma={algorithm.gamma:.8g}, "
        f"iterations={algorithm.n_iter_}, rel_error={rel_error:.5f}, "
        f"sparsity={sparsity:.4f}"
    )


def run():
    print("=" * 72)
    print("Step 4 - Final Optimal Robust Coefficient Matrices")
    print("=" * 72)
    print(f"Input PCA matrix: {INPUT_CSV}")

    X_rows, image_ids, feature_names = load_feature_matrix(INPUT_CSV)
    print(f"Samples : {X_rows.shape[0]}")
    print(f"Features: {X_rows.shape[1]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for method in COEFFICIENT_METHODS:
        print("\n" + "-" * 72)
        print(f"Fitting {method}")
        print("-" * 72)
        algorithm = build_algorithm(method)
        C = algorithm.compute_coefficients(X_rows)
        save_model_outputs(
            method, algorithm, C, X_rows, image_ids, feature_names
        )

    print("\nStep 4 complete. Both final coefficient matrices are ready for Step 5.")
    print("Use:")
    print(f"  {OUTPUT_ROOT / 'robust_ssc' / 'coefficients.csv'}")
    print(f"  {OUTPUT_ROOT / 'robust_lrsc' / 'coefficients.csv'}")


if __name__ == "__main__":
    run()
