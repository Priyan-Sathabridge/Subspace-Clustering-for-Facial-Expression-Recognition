"""
Regenerate improved static and interactive 3D Robust-LRSC surfaces from an
existing lambda_gamma_grid_results.csv file.

This is useful when you already ran the grid search and only want better plots.

Edit INPUT_RESULTS and OUTPUT_FOLDER below, then run:
    python regenerate_interactive_surfaces.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

INPUT_RESULTS = "/Users/priyansathabridge/Desktop/Parameter_Estimation/robust_lrsc_grid/lambda_gamma_grid_results.csv"
OUTPUT_FOLDER = "/Users/priyansathabridge/Desktop/Parameter_Estimation/robust_lrsc_grid"
COLOR_SCALE = "Viridis"     # alternatives: Plasma, Cividis, Turbo
BEST_POINT_REQUIRES_CONVERGENCE = True
SAVE_STATIC_SURFACES = True
SAVE_INTERACTIVE_SURFACES = True
VIEW_ELEVATION = 28
VIEW_AZIMUTH = -132
HTML_INCLUDE_PLOTLYJS = True


def _best_rows(results_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    valid = results_df.dropna(subset=[metric]).copy()
    if BEST_POINT_REQUIRES_CONVERGENCE and "Converged" in valid.columns:
        converged = valid[valid["Converged"] == True]
        if not converged.empty:
            return converged
    return valid


def prepare_surface_data(results_df: pd.DataFrame, metric: str):
    lambda_values = np.sort(results_df["lambda"].dropna().unique().astype(float))
    gamma_values = np.sort(results_df["gamma"].dropna().unique().astype(float))

    metric_grid = results_df.pivot(index="gamma", columns="lambda", values=metric)
    metric_grid = metric_grid.reindex(index=gamma_values, columns=lambda_values)

    conv_grid = results_df.pivot(index="gamma", columns="lambda", values="Converged")
    conv_grid = conv_grid.reindex(index=gamma_values, columns=lambda_values)

    relerr_grid = None
    if "Relative_Error" in results_df.columns:
        relerr_grid = results_df.pivot(index="gamma", columns="lambda", values="Relative_Error")
        relerr_grid = relerr_grid.reindex(index=gamma_values, columns=lambda_values)

    x_log = np.log10(lambda_values)
    y_log = np.log10(gamma_values)
    Xg, Yg = np.meshgrid(x_log, y_log)
    Z = metric_grid.to_numpy(dtype=float)

    return {
        "metric_grid": metric_grid,
        "conv_grid": conv_grid,
        "relerr_grid": relerr_grid,
        "lambda_vals": lambda_values,
        "gamma_vals": gamma_values,
        "Xg": Xg,
        "Yg": Yg,
        "Z": Z,
    }


def plot_static(results_df: pd.DataFrame, metric: str, title: str, output_folder: Path):
    data = prepare_surface_data(results_df, metric)
    Z = np.ma.masked_invalid(data["Z"])

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        data["Xg"], data["Yg"], Z,
        cmap="viridis",
        linewidth=0.2,
        edgecolor="none",
        antialiased=True,
        alpha=0.96,
    )

    valid = _best_rows(results_df, metric)
    if not valid.empty:
        best = valid.loc[valid[metric].idxmax()]
        ax.scatter(np.log10(best["lambda"]), np.log10(best["gamma"]), best[metric], s=85, color="red", depthshade=False)
        ax.text(np.log10(best["lambda"]), np.log10(best["gamma"]), best[metric], f" best={best[metric]:.3f}", fontsize=10)

    ax.set_xlabel(r"$\log_{10}(\lambda)$", labelpad=12, fontsize=11)
    ax.set_ylabel(r"$\log_{10}(\gamma)$", labelpad=12, fontsize=11)
    ax.set_zlabel(metric, labelpad=10, fontsize=11)
    ax.set_title(title, fontsize=13, pad=18)
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    ax.tick_params(labelsize=9)
    if metric in {"NMI", "Purity"}:
        ax.set_zlim(0, 1)

    cbar = fig.colorbar(surf, shrink=0.70, aspect=18, pad=0.08)
    cbar.set_label(metric, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_folder / f"{metric.lower()}_lambda_gamma_surface.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_interactive(results_df: pd.DataFrame, metric: str, title: str, output_folder: Path):
    data = prepare_surface_data(results_df, metric)
    metric_grid = data["metric_grid"]
    conv_grid = data["conv_grid"]
    relerr_grid = data["relerr_grid"]
    Z = data["Z"]

    lambda_matrix, gamma_matrix = np.meshgrid(data["lambda_vals"], data["gamma_vals"])
    conv_num = conv_grid.fillna(False).astype(int).to_numpy()
    relerr = relerr_grid.to_numpy(dtype=float) if relerr_grid is not None else np.full_like(Z, np.nan, dtype=float)
    customdata = np.stack([lambda_matrix, gamma_matrix, conv_num, relerr], axis=-1)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=data["Xg"],
            y=data["Yg"],
            z=Z,
            customdata=customdata,
            colorscale=COLOR_SCALE,
            colorbar=dict(title=metric, len=0.80),
            contours={"z": dict(show=True, usecolormap=True, project_z=True, highlightcolor="white")},
            hovertemplate=(
                "lambda=%{customdata[0]:.6g}<br>"
                "gamma=%{customdata[1]:.6g}<br>"
                f"{metric}=%{{z:.4f}}<br>"
                "Converged=%{customdata[2]:.0f}<br>"
                "Relative_Error=%{customdata[3]:.4f}"
                "<extra></extra>"
            ),
        )
    )

    sampled = results_df.dropna(subset=[metric]).copy()
    if not sampled.empty:
        fig.add_trace(
            go.Scatter3d(
                x=np.log10(sampled["lambda"].to_numpy(dtype=float)),
                y=np.log10(sampled["gamma"].to_numpy(dtype=float)),
                z=sampled[metric].to_numpy(dtype=float),
                mode="markers",
                marker=dict(size=4, color="black", opacity=0.65),
                text=[
                    f"lambda={r['lambda']:.6g}<br>gamma={r['gamma']:.6g}<br>{metric}={r[metric]:.4f}<br>Converged={bool(r.get('Converged', False))}<br>Relative_Error={r.get('Relative_Error', np.nan):.4f}"
                    for _, r in sampled.iterrows()
                ],
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
                name="Best point",
            )
        )

    subtitle = "Best point selected among converged solutions" if BEST_POINT_REQUIRES_CONVERGENCE else "Best point selected among all valid solutions"
    zaxis = dict(title=metric, backgroundcolor="rgba(240,240,240,0.55)", gridcolor="white", zerolinecolor="white")
    if metric in {"NMI", "Purity"}:
        zaxis["range"] = [0, 1]

    fig.update_layout(
        title={"text": f"{title}<br><sup>{subtitle}</sup>", "x": 0.5, "xanchor": "center"},
        width=1100,
        height=800,
        margin=dict(l=0, r=0, t=90, b=0),
        scene=dict(
            xaxis=dict(title="log10(lambda)", backgroundcolor="rgba(240,240,240,0.55)", gridcolor="white", zerolinecolor="white"),
            yaxis=dict(title="log10(gamma)", backgroundcolor="rgba(240,240,240,0.55)", gridcolor="white", zerolinecolor="white"),
            zaxis=zaxis,
            camera=dict(eye=dict(x=1.65, y=1.55, z=0.95)),
            aspectmode="cube",
        ),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.70)"),
    )

    fig.write_html(output_folder / f"{metric.lower()}_lambda_gamma_surface_interactive.html", include_plotlyjs=HTML_INCLUDE_PLOTLYJS, full_html=True)


def main():
    results = pd.read_csv(INPUT_RESULTS)
    output_root = Path(OUTPUT_FOLDER)
    static_folder = output_root / "3d_surfaces"
    interactive_folder = output_root / "3d_surfaces_interactive"
    if SAVE_STATIC_SURFACES:
        static_folder.mkdir(parents=True, exist_ok=True)
    if SAVE_INTERACTIVE_SURFACES:
        interactive_folder.mkdir(parents=True, exist_ok=True)

    for metric, title in [
        ("ARI", "Adjusted Rand Index across Robust LRSC Parameters"),
        ("NMI", "Normalized Mutual Information across Robust LRSC Parameters"),
        ("Purity", "Clustering Purity across Robust LRSC Parameters"),
    ]:
        if SAVE_STATIC_SURFACES:
            plot_static(results, metric, title, static_folder)
        if SAVE_INTERACTIVE_SURFACES:
            plot_interactive(results, metric, title, interactive_folder)

    print(f"Plots saved under: {output_root}")


if __name__ == "__main__":
    main()
