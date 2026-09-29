"""
Two-parameter grid search for Robust Low-Rank Subspace Clustering (Robust LRSC).

For each (lambda, gamma) pair the script:
    1. fits Robust LRSC,
    2. constructs W = |C| + |C|^T,
    3. performs spectral clustering,
    4. computes ARI, NMI and Purity against the labelled expressions,
    5. records convergence / decomposition diagnostics,
    6. saves a master CSV, and
    7. creates THREE 3D response surfaces: ARI, NMI and Purity.

Important
---------
The external labels are being used for parameter selection in this script.
Therefore the resulting best (lambda, gamma) pair is validation/tuning output,
not an unbiased final external test result. Use a separate holdout set if a
final generalisation estimate is required.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from step11_parameter_search.coefficient.robust_low_rank import (
    RobustLowRankSubspaceClustering,
)
from step11_parameter_search.affinity.symmetric import SymmetricAffinity
from step11_parameter_search.clustering.spectral_clustering import (
    SpectralClusteringMethod,
)
from step11_parameter_search.evaluation.compute_metrics import compute_metrics


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = (
    "/Users/priyansathabridge/Desktop/Labelled_Expression_Data/"
    "step_3/pca_features.csv"
)
GROUND_TRUTH_FILE = (
    "/Users/priyansathabridge/Desktop/Labelled_Expression_Data/ground_truth.csv"
)
OUTPUT_FOLDER = (
    "/Users/priyansathabridge/Desktop/Parameter_Estimation/robust_lrsc_grid"
)

# Coarse logarithmic two-parameter grid.
# 10 x 10 = 100 model fits. Refine around a promising region afterwards.
LAMBDA_VALUES = np.logspace(-4, 1, 10)
GAMMA_VALUES = np.logspace(-4, 1, 10)

N_CLUSTERS = 4
RANDOM_STATE = 42

# Robust LRSC / ADMM settings
MAX_ITER = 500
TOL = 1e-6
MU = 1.0
RHO = 1.5
MAX_MU = 1e6
ZERO_DIAGONAL = True

# Set True only if you want each C, W, E and cluster vector saved for every
# grid point. The master results table is always saved.
SAVE_PER_RUN_OUTPUTS = False

# Standard LRSC in the repository normalises each sample before estimating C.
# Leave True for a like-for-like comparison. Set False if the PCA-score scale
# itself is intentionally part of the robust model.
L2_NORMALIZE_SAMPLES = True

# Numerical rank tolerance for diagnostics.
RANK_REL_TOL = 1e-6


# ============================================================
# DATA LOADING AND ALIGNMENT
# ============================================================

NON_FEATURE_COLUMNS = {
    "Image", "image", "Label", "label", "Path", "path", "Cluster", "cluster"
}


def _label_column(df: pd.DataFrame) -> str:
    if "Label" in df.columns:
        return "Label"
    if "label" in df.columns:
        return "label"
    raise ValueError("Ground-truth file must contain a 'Label' column.")


def _image_column(df: pd.DataFrame) -> str | None:
    if "Image" in df.columns:
        return "Image"
    if "image" in df.columns:
        return "image"
    return None


def load_aligned_data(feature_file: str, ground_truth_file: str):
    """
    Load numerical features and true labels.

    If both files contain an Image column, alignment is performed by image
    identifier instead of relying on row order. Otherwise a row-order fallback
    is used after a strict length check.

    Returns
    -------
    X_rows : ndarray, shape (n_samples, n_features)
    labels_true : ndarray, shape (n_samples,)
    image_ids : ndarray, shape (n_samples,)
    feature_names : list[str]
    """

    feature_df = pd.read_csv(feature_file)
    truth_df = pd.read_csv(ground_truth_file)

    label_col = _label_column(truth_df)
    feature_image_col = _image_column(feature_df)
    truth_image_col = _image_column(truth_df)

    if feature_image_col is not None and truth_image_col is not None:
        if feature_df[feature_image_col].duplicated().any():
            raise ValueError("Duplicate image identifiers found in feature file.")
        if truth_df[truth_image_col].duplicated().any():
            raise ValueError("Duplicate image identifiers found in ground-truth file.")

        truth_small = truth_df[[truth_image_col, label_col]].copy()
        if truth_image_col != feature_image_col:
            truth_small = truth_small.rename(columns={truth_image_col: feature_image_col})

        merged = feature_df.merge(
            truth_small,
            on=feature_image_col,
            how="inner",
            validate="one_to_one",
        )

        if len(merged) != len(feature_df) or len(merged) != len(truth_df):
            missing_features = set(truth_df[truth_image_col]) - set(feature_df[feature_image_col])
            missing_truth = set(feature_df[feature_image_col]) - set(truth_df[truth_image_col])
            raise ValueError(
                "Feature and ground-truth image identifiers do not match exactly. "
                f"Missing from features: {len(missing_features)}; "
                f"missing from ground truth: {len(missing_truth)}."
            )

        image_ids = merged[feature_image_col].astype(str).to_numpy()
        labels_true = merged[label_col].to_numpy()

        feature_candidates = merged.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in merged.columns],
            errors="ignore",
        )
    else:
        warnings.warn(
            "Image identifier not present in both files; falling back to row-order "
            "alignment. Add an Image column to both files for safer evaluation.",
            RuntimeWarning,
        )
        if len(feature_df) != len(truth_df):
            raise ValueError(
                "Number of feature rows does not match number of ground-truth rows."
            )
        labels_true = truth_df[label_col].to_numpy()
        image_ids = np.arange(len(feature_df)).astype(str)
        feature_candidates = feature_df.drop(
            columns=[c for c in NON_FEATURE_COLUMNS if c in feature_df.columns],
            errors="ignore",
        )

    feature_candidates = feature_candidates.select_dtypes(include=[np.number])

    if feature_candidates.shape[1] == 0:
        raise ValueError("No numerical feature columns were found.")
    if feature_candidates.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    X_rows = feature_candidates.to_numpy(dtype=float)

    if not np.isfinite(X_rows).all():
        raise ValueError("Feature matrix contains non-finite values.")

    if L2_NORMALIZE_SAMPLES:
        X_rows = normalize(X_rows, norm="l2", axis=1)

    unique_labels = np.unique(labels_true)
    if len(unique_labels) != N_CLUSTERS:
        warnings.warn(
            f"Ground truth contains {len(unique_labels)} unique labels but "
            f"N_CLUSTERS={N_CLUSTERS}.",
            RuntimeWarning,
        )

    print(f"Aligned feature matrix: {X_rows.shape}")
    print(f"Ground-truth labels:     {len(labels_true)}")
    print(f"Expression classes:      {list(unique_labels)}")

    return X_rows, labels_true, image_ids, list(feature_candidates.columns)


# ============================================================
# DIAGNOSTICS
# ============================================================


def numerical_rank(C: np.ndarray, rel_tol: float = RANK_REL_TOL) -> int:
    s = np.linalg.svd(C, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0
    return int(np.sum(s > rel_tol * s[0]))


def decomposition_diagnostics(X: np.ndarray, C: np.ndarray, E: np.ndarray) -> dict:
    """X uses the robust-LRSC convention d x n."""
    eps = np.finfo(float).eps
    reconstruction = X - X @ C - E
    x_norm = max(np.linalg.norm(X, ord="fro"), eps)

    return {
        "Reconstruction_Fro": float(np.linalg.norm(reconstruction, ord="fro")),
        "Relative_Reconstruction": float(
            np.linalg.norm(reconstruction, ord="fro") / x_norm
        ),
        "Error_L1": float(np.sum(np.abs(E))),
        "Error_Fro": float(np.linalg.norm(E, ord="fro")),
        "Relative_Error": float(np.linalg.norm(E, ord="fro") / x_norm),
        "Rank_C": numerical_rank(C),
    }


# ============================================================
# ONE GRID POINT
# ============================================================


def run_single_pair(X_rows: np.ndarray, labels_true: np.ndarray, lam: float, gamma: float):
    """Run Robust LRSC -> affinity -> spectral clustering -> evaluation."""

    # RobustLowRankSubspaceClustering expects X = d x n.
    X = X_rows.T

    model = RobustLowRankSubspaceClustering(
        lambda_reg=float(lam),
        gamma=float(gamma),
        mu=MU,
        rho=RHO,
        max_mu=MAX_MU,
        max_iter=MAX_ITER,
        tol=TOL,
        zero_diagonal=ZERO_DIAGONAL,
        verbose=False,
    )
    model.fit(X)

    C = model.coef_
    E = model.error_

    if C is None or E is None:
        raise RuntimeError("Robust LRSC returned no coefficient/error matrix.")
    if not np.isfinite(C).all() or not np.isfinite(E).all():
        raise FloatingPointError("Robust LRSC produced non-finite values.")

    W = SymmetricAffinity(normalize=True).construct_affinity(C)

    if np.max(W) <= 0 or np.sum(W) <= 0:
        raise ValueError("Affinity matrix is all zero; spectral clustering is undefined.")

    labels_pred = SpectralClusteringMethod(
        n_clusters=N_CLUSTERS,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    ).cluster(W)

    metrics, contingency = compute_metrics(labels_true, labels_pred)
    diagnostics = decomposition_diagnostics(X, C, E)

    diagnostics.update(
        {
            "Iterations": int(model.n_iter_),
            "Converged": bool(getattr(model, "converged_", model.n_iter_ < MAX_ITER)),
            "Final_Constraint_Error": float(
                getattr(model, "constraint_error_", np.nan)
            ),
            "Final_Parameter_Change": float(
                getattr(model, "parameter_change_", np.nan)
            ),
        }
    )

    return C, E, W, labels_pred, metrics, contingency, diagnostics


# ============================================================
# OUTPUT HELPERS
# ============================================================


def safe_pair_name(lam: float, gamma: float) -> str:
    return f"lambda_{lam:.3e}_gamma_{gamma:.3e}".replace("+", "")


def save_per_run(
    root: Path,
    lam: float,
    gamma: float,
    C: np.ndarray,
    E: np.ndarray,
    W: np.ndarray,
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    image_ids: np.ndarray,
    contingency: pd.DataFrame,
):
    folder = root / "grid_runs" / safe_pair_name(lam, gamma)
    folder.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(C).to_csv(folder / "coefficients.csv", index=False)
    pd.DataFrame(E).to_csv(folder / "error_matrix.csv", index=False)
    pd.DataFrame(W).to_csv(folder / "affinity.csv", index=False)
    pd.DataFrame(
        {
            "Image": image_ids,
            "True_Label": labels_true,
            "Cluster": labels_pred,
        }
    ).to_csv(folder / "clusters.csv", index=False)
    contingency.to_csv(folder / "contingency_table.csv")


def save_best_parameter_table(results_df: pd.DataFrame, output_folder: Path):
    rows = []
    for metric in ["ARI", "NMI", "Purity"]:
        valid = results_df.dropna(subset=[metric])
        if valid.empty:
            continue
        row = valid.loc[valid[metric].idxmax()].copy()
        rows.append(
            {
                "Criterion": metric,
                "lambda": row["lambda"],
                "gamma": row["gamma"],
                metric: row[metric],
                "ARI": row["ARI"],
                "NMI": row["NMI"],
                "Purity": row["Purity"],
                "Converged": row.get("Converged", np.nan),
                "Relative_Reconstruction": row.get("Relative_Reconstruction", np.nan),
                "Relative_Error": row.get("Relative_Error", np.nan),
                "Rank_C": row.get("Rank_C", np.nan),
            }
        )

    pd.DataFrame(rows).to_csv(output_folder / "best_parameters_by_metric.csv", index=False)


# ============================================================
# THREE 3D EVALUATION SURFACES
# ============================================================


def plot_metric_surface(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    output_folder: Path,
):
    """
    Save one 3D response surface with axes log10(lambda), log10(gamma), metric.

    Missing/failed grid points are masked rather than interpolated.
    """

    pivot = results_df.pivot(index="gamma", columns="lambda", values=metric)
    pivot = pivot.reindex(index=GAMMA_VALUES, columns=LAMBDA_VALUES)

    lambda_grid, gamma_grid = np.meshgrid(
        pivot.columns.to_numpy(dtype=float),
        pivot.index.to_numpy(dtype=float),
    )
    z = np.ma.masked_invalid(pivot.to_numpy(dtype=float))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        np.log10(lambda_grid),
        np.log10(gamma_grid),
        z,
        rstride=1,
        cstride=1,
        linewidth=0.25,
        antialiased=True,
    )

    valid = results_df.dropna(subset=[metric])
    if not valid.empty:
        best = valid.loc[valid[metric].idxmax()]
        ax.scatter(
            np.log10(best["lambda"]),
            np.log10(best["gamma"]),
            best[metric],
            s=55,
            marker="o",
        )
        ax.text(
            np.log10(best["lambda"]),
            np.log10(best["gamma"]),
            best[metric],
            f"  best={best[metric]:.3f}",
        )
    else:
        ax.text2D(
            0.5, 0.5,
            f"No valid {metric} values were produced.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel(r"$\log_{10}(\lambda)$")
    ax.set_ylabel(r"$\log_{10}(\gamma)$")
    ax.set_zlabel(metric)
    ax.set_title(title)

    # External metrics all lie in bounded ranges; keeping the native range
    # preserves negative ARI values if they occur.
    if metric in {"NMI", "Purity"}:
        ax.set_zlim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(
        output_folder / f"{metric.lower()}_lambda_gamma_surface.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_all_surfaces(results_df: pd.DataFrame, output_folder: Path):
    surface_folder = output_folder / "3d_surfaces"
    surface_folder.mkdir(parents=True, exist_ok=True)

    specs = [
        ("ARI", "Adjusted Rand Index across Robust LRSC Parameters"),
        ("NMI", "Normalized Mutual Information across Robust LRSC Parameters"),
        ("Purity", "Clustering Purity across Robust LRSC Parameters"),
    ]

    grid_folder = output_folder / "metric_grids"
    grid_folder.mkdir(parents=True, exist_ok=True)

    for metric, title in specs:
        grid = results_df.pivot(index="gamma", columns="lambda", values=metric)
        grid = grid.reindex(index=GAMMA_VALUES, columns=LAMBDA_VALUES)
        grid.to_csv(grid_folder / f"{metric.lower()}_grid.csv")
        plot_metric_surface(results_df, metric, title, surface_folder)

    print(f"3D evaluation surfaces saved to: {surface_folder}")


# ============================================================
# FULL TWO-PARAMETER GRID SEARCH
# ============================================================


def run_grid_search():
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    X_rows, labels_true, image_ids, feature_names = load_aligned_data(
        FEATURE_FILE, GROUND_TRUTH_FILE
    )

    pd.DataFrame(
        {
            "Parameter": [
                "n_samples",
                "n_features",
                "n_clusters",
                "max_iter",
                "tol",
                "mu",
                "rho",
                "max_mu",
                "l2_normalize_samples",
            ],
            "Value": [
                X_rows.shape[0],
                X_rows.shape[1],
                N_CLUSTERS,
                MAX_ITER,
                TOL,
                MU,
                RHO,
                MAX_MU,
                L2_NORMALIZE_SAMPLES,
            ],
        }
    ).to_csv(output_folder / "grid_search_configuration.csv", index=False)
    pd.DataFrame({"Feature": feature_names}).to_csv(
        output_folder / "features_used.csv", index=False
    )

    all_results = []
    total = len(LAMBDA_VALUES) * len(GAMMA_VALUES)
    count = 0

    for lam in LAMBDA_VALUES:
        for gamma in GAMMA_VALUES:
            count += 1
            print("\n" + "=" * 72)
            print(
                f"Grid point {count}/{total}: "
                f"lambda={lam:.6g}, gamma={gamma:.6g}"
            )
            print("=" * 72)

            result = {
                "lambda": float(lam),
                "gamma": float(gamma),
                "ARI": np.nan,
                "NMI": np.nan,
                "Purity": np.nan,
                "Converged": False,
                "Error": "",
            }

            try:
                (
                    C,
                    E,
                    W,
                    labels_pred,
                    metrics,
                    contingency,
                    diagnostics,
                ) = run_single_pair(X_rows, labels_true, lam, gamma)

                result.update(metrics)
                result.update(diagnostics)

                print(
                    f"ARI={metrics['ARI']:.4f} | "
                    f"NMI={metrics['NMI']:.4f} | "
                    f"Purity={metrics['Purity']:.4f} | "
                    f"Converged={result['Converged']} | "
                    f"Iter={result['Iterations']}"
                )

                if SAVE_PER_RUN_OUTPUTS:
                    save_per_run(
                        output_folder,
                        lam,
                        gamma,
                        C,
                        E,
                        W,
                        labels_pred,
                        labels_true,
                        image_ids,
                        contingency,
                    )

            except Exception as exc:
                result["Error"] = f"{type(exc).__name__}: {exc}"
                print(f"ERROR: {result['Error']}")

            all_results.append(result)

            # Incremental checkpoint so a long grid is recoverable.
            pd.DataFrame(all_results).to_csv(
                output_folder / "lambda_gamma_grid_results_checkpoint.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["lambda", "gamma"]).reset_index(drop=True)
    results_df.to_csv(output_folder / "lambda_gamma_grid_results.csv", index=False)

    save_best_parameter_table(results_df, output_folder)
    plot_all_surfaces(results_df, output_folder)

    # Compact console summary.
    print("\n" + "=" * 72)
    print("ROBUST LRSC TWO-PARAMETER GRID SEARCH COMPLETE")
    print("=" * 72)

    successful = results_df.dropna(subset=["ARI", "NMI", "Purity"])
    print(f"Successful runs: {len(successful)}/{len(results_df)}")

    for metric in ["ARI", "NMI", "Purity"]:
        valid = results_df.dropna(subset=[metric])
        if valid.empty:
            print(f"No valid result for {metric}.")
            continue
        best = valid.loc[valid[metric].idxmax()]
        print(
            f"Best {metric}: {best[metric]:.4f} at "
            f"lambda={best['lambda']:.6g}, gamma={best['gamma']:.6g}"
        )

    return results_df


if __name__ == "__main__":
    results = run_grid_search()
    print("\nComplete results:\n")
    print(results.to_string(index=False))
