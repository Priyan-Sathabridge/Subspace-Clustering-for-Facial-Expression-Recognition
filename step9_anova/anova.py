# anova.py

import os
import numpy as np
import pandas as pd

from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests


def eta_squared(groups):
    """
    Computes eta-squared for one-way ANOVA.
    """

    all_values = np.concatenate(groups)

    grand_mean = np.mean(all_values)

    ss_between = sum(
        len(g) * (np.mean(g) - grand_mean) ** 2
        for g in groups
    )

    ss_total = np.sum((all_values - grand_mean) ** 2)

    return ss_between / ss_total if ss_total > 0 else np.nan


def run_anova(
    feature_csv,
    cluster_csv,
    output_csv,
    image_column="image",
    cluster_column="cluster"
):
    """
    Run one-way ANOVA on every feature.
    """

    features = pd.read_csv(feature_csv)
    clusters = pd.read_csv(cluster_csv)

    data = features.merge(
        clusters[[image_column, cluster_column]],
        on=image_column
    )

    feature_columns = [
        c for c in data.columns
        if c not in [image_column, cluster_column]
    ]

    results = []

    for feature in feature_columns:

        grouped = []

        for cluster in sorted(data[cluster_column].unique()):

            values = data.loc[
                data[cluster_column] == cluster,
                feature
            ].dropna().values

            if len(values) > 1:
                grouped.append(values)

        if len(grouped) < 2:
            continue

        F, p = f_oneway(*grouped)

        eta = eta_squared(grouped)

        results.append({
            "Feature": feature,
            "F Statistic": F,
            "P Value": p,
            "Eta Squared": eta
        })

    results = pd.DataFrame(results)

    # Control false discovery rate across the feature-wise tests. These are
    # descriptive comparisons of clusters learned from the same feature data.
    reject, p_adj, _, _ = multipletests(
        results["P Value"],
        method="fdr_bh"
    )

    results["Adjusted P Value"] = p_adj
    results["Significant"] = reject

    results = results.sort_values(
        "Eta Squared",
        ascending=False
    )

    os.makedirs(
        os.path.dirname(output_csv),
        exist_ok=True
    )

    results.to_csv(
        output_csv,
        index=False
    )

    print(results.head(20))

    print(f"\nResults saved to:\n{output_csv}")


if __name__ == "__main__":

    run_anova(
        feature_csv="/Users/priyansathabridge/Desktop/Sparse_Results/inputs/pca_features.csv",
        cluster_csv="/Users/priyansathabridge/Desktop/Sparse_Results/outputs/k_20/labels.csv",
        output_csv="/Users/priyansathabridge/Desktop/Sparse_Results/outputs/k_20/anova_pca.csv",
        image_column="Image",
        cluster_column="Cluster"
    )