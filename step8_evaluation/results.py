"""
results.py

Utilities for:
    - Combining evaluation metrics
    - Saving results
    - Plotting evaluation metrics

Author: Priyan Sathabridge
"""

import os

import matplotlib.pyplot as plt

import pandas as pd


# =========================================================
# Combine Results
# =========================================================

def combine_results(metric_results):
    """
    Combine all evaluation metrics into a single DataFrame.

    Parameters
    ----------
    metric_results : dict
        Dictionary where keys are metric names and values are
        dictionaries of the form:

        {
            "Silhouette": {4:0.22, 5:0.30, ...},
            "DBI": {4:1.10, 5:0.95, ...},
            ...
        }

    Returns
    -------
    pandas.DataFrame
    """

    # Determine every k that appears
    all_k = sorted({

        k

        for metric in metric_results.values()

        for k in metric.keys()

    })

    df = pd.DataFrame({

        "k": all_k

    })

    # Add each metric as a column
    for metric_name, values in metric_results.items():

        df[metric_name] = [

            values.get(k)

            for k in all_k

        ]

    return df


# =========================================================
# Save Results
# =========================================================

def save_results(results_df,
                 output_path):
    """
    Save evaluation results to CSV.
    """

    os.makedirs(

        os.path.dirname(output_path),

        exist_ok=True

    )

    results_df.to_csv(

        output_path,

        index=False

    )

    print(

        f"Saved evaluation results to:\n{output_path}"

    )


# =========================================================
# Plot One Metric
# =========================================================

def plot_metric(results_df,
                metric,
                output_folder):
    """
    Plot a single evaluation metric.
    """

    if metric not in results_df.columns:

        return

    df = results_df.dropna(

        subset=[metric]

    )

    if df.empty:

        return

    plt.figure(figsize=(7,5))

    plt.plot(

        df["k"],

        df[metric],

        marker="o",

        linewidth=2

    )

    plt.xlabel("Number of Clusters (k)")

    plt.ylabel(metric)

    plt.title(metric)

    plt.grid(True)

    filename = (

        metric

        .lower()

        .replace(" ", "_")

        + ".png"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            output_folder,

            filename

        )

    )

    plt.close()


# =========================================================
# Plot All Metrics
# =========================================================

def plot_all_results(results_df,
                     output_folder):
    """
    Generate one plot for every metric.
    """

    os.makedirs(

        output_folder,

        exist_ok=True

    )

    for column in results_df.columns:

        if column == "k":

            continue

        plot_metric(

            results_df,

            column,

            output_folder

        )

    print(

        f"Saved plots to:\n{output_folder}"

    )