"""
Two-parameter grid search for Robust Sparse Subspace Clustering (Robust SSC).

The formulation is deliberately matched to the Robust LRSC experiment:

    Robust LRSC:
        min lambda ||C||_* + gamma ||E||_1
        s.t. X = X C + E

    Robust SSC:
        min lambda ||C||_1 + gamma ||E||_1
        s.t. X = X C + E, diag(C) = 0

Thus the downstream pipeline is identical; only the structural regulariser on C
changes from the nuclear norm (low rank) to the entrywise L1 norm (sparsity).

For every (lambda, gamma) pair this script:
    1. fits Robust SSC with ADMM,
    2. constructs W = |C| + |C|^T,
    3. performs spectral clustering,
    4. computes ARI, NMI and Purity,
    5. records convergence, error and coefficient-sparsity diagnostics,
    6. saves a master CSV and best-parameter table,
    7. creates static colour-gradient 3D surfaces, and
    8. creates interactive Plotly HTML surfaces.

Ground-truth labels are used for parameter tuning in this script. The resulting
best parameter pair is therefore validation/tuning output rather than an
independent external test estimate.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import normalize

from step11_parameter_search.coefficient.robust_sparse import (
    RobustSparseSubspaceClustering,
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
    "/Users/priyansathabridge/Desktop/Parameter_Estimation/robust_ssc_grid_1"
)

# Coarse first-round grid. This deliberately mirrors the first Robust LRSC
# search. Change these ranges later if a second-round refinement is required.
LAMBDA_VALUES = np.logspace(
    np.log10(0.005),
    np.log10(0.05),
    15
)

GAMMA_VALUES = np.logspace(
    np.log10(0.20),
    np.log10(1.00),
    15
)

N_CLUSTERS = 4
RANDOM_STATE = 42

# Robust SSC / ADMM settings. These match the Robust LRSC settings so that the
# optimisation frameworks remain comparable.
MAX_ITER = 1000
TOL = 1e-6
MU = 1.0
RHO = 1.5
MAX_MU = 1e6
ZERO_DIAGONAL = True

SAVE_PER_RUN_OUTPUTS = False
L2_NORMALIZE_SAMPLES = True

# A coefficient is treated as structurally active if its magnitude exceeds
# max(COEFFICIENT_ABS_TOL, COEFFICIENT_REL_TOL * max(|C|)).
COEFFICIENT_REL_TOL = 1e-6
COEFFICIENT_ABS_TOL = 1e-10

# Plot settings
SAVE_STATIC_SURFACES = True
SAVE_INTERACTIVE_SURFACES = True
COLOR_SCALE = "Viridis"
BEST_POINT_REQUIRES_CONVERGENCE = True
HTML_INCLUDE_PLOTLYJS = True
VIEW_ELEVATION = 28
VIEW_AZIMUTH = -132

NON_FEATURE_COLUMNS = {
    "Image", "image", "Label", "label", "Path", "path", "Cluster", "cluster"
}


# ============================================================
# DATA LOADING AND ALIGNMENT
# ============================================================


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
    """Load numerical features and align labels by Image whenever possible."""
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

    # Same sample-wise L2 normalization used by the Robust LRSC parameter search.
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


def coefficient_structure(C: np.ndarray) -> dict:
    """Quantify sparsity of the learned self-representation matrix."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square.")

    max_abs = float(np.max(np.abs(C))) if C.size else 0.0
    threshold = max(COEFFICIENT_ABS_TOL, COEFFICIENT_REL_TOL * max_abs)

    active = np.abs(C) > threshold
    np.fill_diagonal(active, False)

    possible = n * (n - 1)
    active_count = int(np.count_nonzero(active))
    density = float(active_count / possible) if possible > 0 else 0.0

    return {
        "Coefficient_L1": float(np.sum(np.abs(C))),
        "Coefficient_Active": active_count,
        "Coefficient_Density": density,
        "Coefficient_Sparsity": float(1.0 - density),
        "Mean_Active_Per_Sample": float(active_count / n) if n > 0 else 0.0,
        "Coefficient_Threshold": float(threshold),
    }


def decomposition_diagnostics(
    X: np.ndarray,
    C: np.ndarray,
    E: np.ndarray,
    J: np.ndarray | None = None,
) -> dict:
    """X follows the d x n convention."""
    eps = np.finfo(float).eps
    reconstruction = X - X @ C - E
    x_norm = max(np.linalg.norm(X, ord="fro"), eps)

    diagnostics = {
        "Reconstruction_Fro": float(np.linalg.norm(reconstruction, ord="fro")),
        "Relative_Reconstruction": float(
            np.linalg.norm(reconstruction, ord="fro") / x_norm
        ),
        "Error_L1": float(np.sum(np.abs(E))),
        "Error_Fro": float(np.linalg.norm(E, ord="fro")),
        "Relative_Error": float(np.linalg.norm(E, ord="fro") / x_norm),
    }
    diagnostics.update(coefficient_structure(C))

    if J is not None:
        aux = coefficient_structure(J)
        diagnostics.update(
            {
                "Auxiliary_Density": aux["Coefficient_Density"],
                "Auxiliary_Sparsity": aux["Coefficient_Sparsity"],
                "Auxiliary_Active": aux["Coefficient_Active"],
            }
        )

    return diagnostics


# ============================================================
# ONE GRID POINT
# ============================================================


def run_single_pair(X_rows: np.ndarray, labels_true: np.ndarray, lam: float, gamma: float):
    """Run Robust SSC -> affinity -> spectral clustering -> external metrics."""
    X = X_rows.T

    model = RobustSparseSubspaceClustering(
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
    J = getattr(model, "auxiliary_", None)

    if C is None or E is None:
        raise RuntimeError("Robust SSC returned no coefficient/error matrix.")
    if not np.isfinite(C).all() or not np.isfinite(E).all():
        raise FloatingPointError("Robust SSC produced non-finite values.")

    W = SymmetricAffinity(normalize=True).construct_affinity(C)
    if np.max(W) <= 0 or np.sum(W) <= 0:
        raise ValueError("Affinity matrix is all zero; spectral clustering is undefined.")

    labels_pred = SpectralClusteringMethod(
        n_clusters=N_CLUSTERS,
        assign_labels="kmeans",
        random_state=RANDOM_STATE,
    ).cluster(W)

    metrics, contingency = compute_metrics(labels_true, labels_pred)
    diagnostics = decomposition_diagnostics(X, C, E, J)
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


def _best_rows(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = results_df.dropna(subset=[metric]).copy()
    if BEST_POINT_REQUIRES_CONVERGENCE and "Converged" in valid.columns:
        converged = valid[valid["Converged"] == True]
        if not converged.empty:
            return converged
    return valid


def save_best_parameter_table(results_df: pd.DataFrame, output_folder: Path):
    rows = []
    for metric in ["ARI", "NMI", "Purity"]:
        valid = _best_rows(results_df, metric)
        if valid.empty:
            continue
        row = valid.loc[valid[metric].idxmax()].copy()
        rows.append(
            {
                "Criterion": metric,
                "lambda": row["lambda"],
                "gamma": row["gamma"],
                "ARI": row["ARI"],
                "NMI": row["NMI"],
                "Purity": row["Purity"],
                "Converged": row.get("Converged", np.nan),
                "Relative_Reconstruction": row.get("Relative_Reconstruction", np.nan),
                "Relative_Error": row.get("Relative_Error", np.nan),
                "Coefficient_Sparsity": row.get("Coefficient_Sparsity", np.nan),
                "Mean_Active_Per_Sample": row.get("Mean_Active_Per_Sample", np.nan),
            }
        )

    pd.DataFrame(rows).to_csv(output_folder / "best_parameters_by_metric.csv", index=False)


# ============================================================
# STATIC + INTERACTIVE 3D SURFACES
# ============================================================


def prepare_surface_data(results_df: pd.DataFrame, metric: str):
    lambda_values = np.sort(results_df["lambda"].dropna().unique().astype(float))
    gamma_values = np.sort(results_df["gamma"].dropna().unique().astype(float))

    metric_grid = results_df.pivot(index="gamma", columns="lambda", values=metric)
    metric_grid = metric_grid.reindex(index=gamma_values, columns=lambda_values)

    conv_grid = results_df.pivot(index="gamma", columns="lambda", values="Converged")
    conv_grid = conv_grid.reindex(index=gamma_values, columns=lambda_values)

    relerr_grid = results_df.pivot(index="gamma", columns="lambda", values="Relative_Error")
    relerr_grid = relerr_grid.reindex(index=gamma_values, columns=lambda_values)

    sparsity_grid = results_df.pivot(
        index="gamma", columns="lambda", values="Coefficient_Sparsity"
    )
    sparsity_grid = sparsity_grid.reindex(index=gamma_values, columns=lambda_values)

    x_log = np.log10(lambda_values)
    y_log = np.log10(gamma_values)
    Xg, Yg = np.meshgrid(x_log, y_log)

    return {
        "metric_grid": metric_grid,
        "conv_grid": conv_grid,
        "relerr_grid": relerr_grid,
        "sparsity_grid": sparsity_grid,
        "lambda_vals": lambda_values,
        "gamma_vals": gamma_values,
        "Xg": Xg,
        "Yg": Yg,
        "Z": metric_grid.to_numpy(dtype=float),
    }


def plot_metric_surface_static(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    output_folder: Path,
):
    data = prepare_surface_data(results_df, metric)
    Z = np.ma.masked_invalid(data["Z"])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        data["Xg"],
        data["Yg"],
        Z,
        cmap="viridis",
        linewidth=0.2,
        edgecolor="none",
        antialiased=True,
        alpha=0.96,
    )

    valid = _best_rows(results_df, metric)
    if not valid.empty:
        best = valid.loc[valid[metric].idxmax()]
        ax.scatter(
            np.log10(best["lambda"]),
            np.log10(best["gamma"]),
            best[metric],
            s=85,
            marker="o",
            color="red",
            depthshade=False,
        )
        ax.text(
            np.log10(best["lambda"]),
            np.log10(best["gamma"]),
            best[metric],
            f" best={best[metric]:.3f}",
            fontsize=10,
        )

    ax.set_xlabel(r"$\log_{10}(\lambda)$", labelpad=12, fontsize=11)
    ax.set_ylabel(r"$\log_{10}(\gamma)$", labelpad=12, fontsize=11)
    ax.set_zlabel(metric, labelpad=10, fontsize=11)
    ax.set_title(title, fontsize=13, pad=18)
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    ax.tick_params(labelsize=9)

    if metric in {"NMI", "Purity"}:
        ax.set_zlim(0.0, 1.0)

    cbar = fig.colorbar(surf, shrink=0.70, aspect=18, pad=0.08)
    cbar.set_label(metric, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(
        output_folder / f"{metric.lower()}_lambda_gamma_surface.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_surface_interactive(
    results_df: pd.DataFrame,
    metric: str,
    title: str,
    output_folder: Path,
):
    data = prepare_surface_data(results_df, metric)
    Z = data["Z"]

    lambda_matrix, gamma_matrix = np.meshgrid(
        data["lambda_vals"], data["gamma_vals"]
    )
    conv_num = data["conv_grid"].fillna(False).astype(int).to_numpy()
    relerr = data["relerr_grid"].to_numpy(dtype=float)
    sparsity = data["sparsity_grid"].to_numpy(dtype=float)

    customdata = np.stack(
        [lambda_matrix, gamma_matrix, conv_num, relerr, sparsity], axis=-1
    )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=data["Xg"],
            y=data["Yg"],
            z=Z,
            customdata=customdata,
            colorscale=COLOR_SCALE,
            colorbar=dict(title=metric, len=0.80),
            contours={
                "z": dict(
                    show=True,
                    usecolormap=True,
                    project_z=True,
                    highlightcolor="white",
                )
            },
            hovertemplate=(
                "lambda=%{customdata[0]:.6g}<br>"
                "gamma=%{customdata[1]:.6g}<br>"
                f"{metric}=%{{z:.4f}}<br>"
                "Converged=%{customdata[2]:.0f}<br>"
                "Relative_Error=%{customdata[3]:.4f}<br>"
                "Coefficient_Sparsity=%{customdata[4]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    sampled = results_df.dropna(subset=[metric]).copy()
    if not sampled.empty:
        marker_text = [
            f"lambda={row['lambda']:.6g}<br>"
            f"gamma={row['gamma']:.6g}<br>"
            f"{metric}={row[metric]:.4f}<br>"
            f"Converged={bool(row.get('Converged', False))}<br>"
            f"Relative_Error={row.get('Relative_Error', np.nan):.4f}<br>"
            f"Coefficient_Sparsity={row.get('Coefficient_Sparsity', np.nan):.4f}"
            for _, row in sampled.iterrows()
        ]
        fig.add_trace(
            go.Scatter3d(
                x=np.log10(sampled["lambda"].to_numpy(dtype=float)),
                y=np.log10(sampled["gamma"].to_numpy(dtype=float)),
                z=sampled[metric].to_numpy(dtype=float),
                mode="markers",
                marker=dict(size=4, color="black", opacity=0.65),
                text=marker_text,
                hovertemplate="%{text}<extra>grid point</extra>",
                name="Sampled grid points",
            )
        )

    valid = _best_rows(results_df, metric)
    if not valid.empty:
        best = valid.loc[valid[metric].idxmax()]
        fig.add_trace(
            go.Scatter3d(
                x=[np.log10(best["lambda"])],
                y=[np.log10(best["gamma"])],
                z=[best[metric]],
                mode="markers+text",
                marker=dict(size=7, color="red", symbol="diamond"),
                text=[f"Best {metric}"],
                textposition="top center",
                hovertemplate=(
                    f"Best {metric}<br>"
                    f"lambda={best['lambda']:.6g}<br>"
                    f"gamma={best['gamma']:.6g}<br>"
                    f"{metric}={best[metric]:.4f}<br>"
                    f"Converged={bool(best.get('Converged', False))}<br>"
                    f"Sparsity={best.get('Coefficient_Sparsity', np.nan):.4f}"
                    "<extra></extra>"
                ),
                name="Best point",
            )
        )

    subtitle = (
        "Best point selected among converged solutions"
        if BEST_POINT_REQUIRES_CONVERGENCE
        else "Best point selected among all valid solutions"
    )

    zaxis = dict(
        title=metric,
        backgroundcolor="rgba(240,240,240,0.55)",
        gridcolor="white",
        zerolinecolor="white",
    )
    if metric in {"NMI", "Purity"}:
        zaxis["range"] = [0, 1]

    fig.update_layout(
        title={
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0.5,
            "xanchor": "center",
        },
        width=1100,
        height=800,
        margin=dict(l=0, r=0, t=90, b=0),
        scene=dict(
            xaxis=dict(
                title="log10(lambda)",
                backgroundcolor="rgba(240,240,240,0.55)",
                gridcolor="white",
                zerolinecolor="white",
            ),
            yaxis=dict(
                title="log10(gamma)",
                backgroundcolor="rgba(240,240,240,0.55)",
                gridcolor="white",
                zerolinecolor="white",
            ),
            zaxis=zaxis,
            camera=dict(eye=dict(x=1.65, y=1.55, z=0.95)),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.70)"),
    )

    fig.write_html(
        output_folder / f"{metric.lower()}_lambda_gamma_surface_interactive.html",
        include_plotlyjs=HTML_INCLUDE_PLOTLYJS,
        full_html=True,
    )


def plot_all_surfaces(results_df: pd.DataFrame, output_folder: Path):
    specs = [
        ("ARI", "Adjusted Rand Index across Robust SSC Parameters"),
        ("NMI", "Normalized Mutual Information across Robust SSC Parameters"),
        ("Purity", "Clustering Purity across Robust SSC Parameters"),
    ]

    grid_folder = output_folder / "metric_grids"
    grid_folder.mkdir(parents=True, exist_ok=True)

    static_folder = output_folder / "3d_surfaces"
    interactive_folder = output_folder / "3d_surfaces_interactive"
    if SAVE_STATIC_SURFACES:
        static_folder.mkdir(parents=True, exist_ok=True)
    if SAVE_INTERACTIVE_SURFACES:
        interactive_folder.mkdir(parents=True, exist_ok=True)

    for metric, title in specs:
        grid = results_df.pivot(index="gamma", columns="lambda", values=metric)
        lambda_values = np.sort(results_df["lambda"].unique().astype(float))
        gamma_values = np.sort(results_df["gamma"].unique().astype(float))
        grid = grid.reindex(index=gamma_values, columns=lambda_values)
        grid.to_csv(grid_folder / f"{metric.lower()}_grid.csv")

        if SAVE_STATIC_SURFACES:
            plot_metric_surface_static(results_df, metric, title, static_folder)
        if SAVE_INTERACTIVE_SURFACES:
            plot_metric_surface_interactive(results_df, metric, title, interactive_folder)

    print(f"Metric grids saved to: {grid_folder}")
    if SAVE_STATIC_SURFACES:
        print(f"Static 3D surfaces saved to: {static_folder}")
    if SAVE_INTERACTIVE_SURFACES:
        print(f"Interactive 3D surfaces saved to: {interactive_folder}")


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
                "model",
                "n_samples",
                "n_features",
                "n_clusters",
                "max_iter",
                "tol",
                "mu",
                "rho",
                "max_mu",
                "l2_normalize_samples",
                "coefficient_rel_tol",
                "plot_color_scale",
                "best_point_requires_convergence",
            ],
            "Value": [
                "Robust SSC: lambda*||C||_1 + gamma*||E||_1",
                X_rows.shape[0],
                X_rows.shape[1],
                N_CLUSTERS,
                MAX_ITER,
                TOL,
                MU,
                RHO,
                MAX_MU,
                L2_NORMALIZE_SAMPLES,
                COEFFICIENT_REL_TOL,
                COLOR_SCALE,
                BEST_POINT_REQUIRES_CONVERGENCE,
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
                    f"Sparsity={result['Coefficient_Sparsity']:.4f} | "
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

            pd.DataFrame(all_results).to_csv(
                output_folder / "lambda_gamma_grid_results_checkpoint.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(["lambda", "gamma"]).reset_index(drop=True)
    results_df.to_csv(output_folder / "lambda_gamma_grid_results.csv", index=False)

    save_best_parameter_table(results_df, output_folder)
    plot_all_surfaces(results_df, output_folder)

    print("\n" + "=" * 72)
    print("ROBUST SSC TWO-PARAMETER GRID SEARCH COMPLETE")
    print("=" * 72)

    successful = results_df.dropna(subset=["ARI", "NMI", "Purity"])
    converged = successful[successful["Converged"] == True]
    print(f"Successful runs: {len(successful)}/{len(results_df)}")
    print(f"Converged runs:  {len(converged)}/{len(results_df)}")

    for metric in ["ARI", "NMI", "Purity"]:
        valid = _best_rows(results_df, metric)
        if valid.empty:
            print(f"No valid result for {metric}.")
            continue
        best = valid.loc[valid[metric].idxmax()]
        print(
            f"Best {metric}: {best[metric]:.4f} at "
            f"lambda={best['lambda']:.6g}, gamma={best['gamma']:.6g} | "
            f"Sparsity={best['Coefficient_Sparsity']:.4f} | "
            f"Converged={bool(best.get('Converged', False))}"
        )

    return results_df


if __name__ == "__main__":
    results = run_grid_search()
    print("\nComplete results:\n")
    print(results.to_string(index=False))
