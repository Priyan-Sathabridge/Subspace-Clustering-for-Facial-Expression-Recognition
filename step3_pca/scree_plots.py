"""
scree_plots.py

Creates an interactive scree plot.

Outputs
-------
scree_plot.html
"""

from pathlib import Path

import plotly.graph_objects as go


def create_scree_plot(
    variance_table,
    variance_threshold,
    output_html,
):
    """
    Create an interactive scree plot.

    Parameters
    ----------
    variance_table : pandas.DataFrame

    variance_threshold : float
        Example: 0.95

    output_html : str or Path
    """

    output_html = Path(output_html)

    explained = variance_table["Explained Variance"]
    cumulative = variance_table["Cumulative Variance"]

    pcs = list(range(1, len(explained) + 1))

    # Number of retained components
    retained = (
        cumulative >= variance_threshold
    ).idxmax() + 1

    fig = go.Figure()

    # Explained variance bars
    fig.add_trace(
        go.Bar(
            x=pcs,
            y=explained,
            name="Explained Variance",
        )
    )

    # Cumulative variance line
    fig.add_trace(
        go.Scatter(
            x=pcs,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative Variance",
        )
    )

    # Horizontal threshold
    fig.add_hline(
        y=variance_threshold,
        line_dash="dash",
        annotation_text=f"{variance_threshold:.0%} Variance",
    )

    # Vertical retained PC
    fig.add_vline(
        x=retained,
        line_dash="dash",
        annotation_text=f"{retained} PCs",
    )

    fig.update_layout(

        title="PCA Scree Plot",

        xaxis_title="Principal Component",

        yaxis_title="Explained Variance",

        template="plotly_white",

        hovermode="x unified",
    )

    fig.write_html(
        output_html,
        include_plotlyjs="cdn"
    )

    print(f"Scree plot saved to:\n{output_html}")