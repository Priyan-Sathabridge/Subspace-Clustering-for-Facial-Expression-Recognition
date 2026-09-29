"""
pca.py

Performs Principal Component Analysis.

Input:
    normalized feature matrix

Outputs:
    - PCA scores
    - loading matrix
    - explained variance table
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(
    X_norm,
    variance_threshold=0.95,
):
    """
    Fit PCA retaining a specified percentage of variance.

    Parameters
    ----------
    X_norm : pandas.DataFrame

    variance_threshold : float
        e.g. 0.95

    Returns
    -------
    pca
    X_pca
    """

    pca = PCA(
        n_components=variance_threshold,
        svd_solver="full",
        random_state=42,
    )

    X_pca = pca.fit_transform(X_norm)

    return pca, X_pca


def get_pca_scores(
    image_names,
    X_pca,
):
    """
    Create PCA score dataframe.
    """

    columns = [
        f"PC{i+1}"
        for i in range(X_pca.shape[1])
    ]

    df_scores = pd.DataFrame(
        X_pca,
        columns=columns,
    )

    df_scores.insert(
        0,
        image_names.name,
        image_names.values,
    )

    return df_scores


def get_loading_matrix(
    pca,
    feature_names,
):
    """
    Compute PCA loading matrix.
    """

    loadings = pca.components_.T

    columns = [
        f"PC{i+1}"
        for i in range(loadings.shape[1])
    ]

    df_loadings = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=columns,
    )

    return df_loadings


def get_variance_table(
    pca,
):
    """
    Create explained variance table.
    """

    explained = pca.explained_variance_ratio_

    cumulative = np.cumsum(explained)

    df = pd.DataFrame(
        {
            "Principal Component":
                [f"PC{i+1}" for i in range(len(explained))],

            "Explained Variance":
                explained,

            "Cumulative Variance":
                cumulative,
        }
    )

    return df


def save_outputs(
    output_folder,
    scores,
    loadings,
    variance_table,
):
    """
    Save all PCA outputs.
    """

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        output_folder / "pca_features.csv",
        index=False,
    )

    loadings.to_csv(
        output_folder / "loadings.csv",
    )

    variance_table.to_csv(
        output_folder / "explained_variance.csv",
        index=False,
    )

    print(f"PCA outputs saved to:\n{output_folder}")