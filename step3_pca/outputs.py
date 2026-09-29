"""
outputs.py

Exports all PCA outputs.
"""

from pathlib import Path

import pandas as pd


def save_normalized_features(
    image_names,
    X_input,
    output_folder,
):
    """
    Save normalized feature matrix.
    """

    output_folder = Path(output_folder)

    df = pd.concat(
        [
            image_names.reset_index(drop=True),
            X_input.reset_index(drop=True),
        ],
        axis=1,
    )

    df.to_csv(
        output_folder / "normalized_features.csv",
        index=False,
    )


def save_pca_scores(
    scores,
    output_folder,
):
    """
    Save PCA feature matrix.
    """

    output_folder = Path(output_folder)

    scores.to_csv(
        output_folder / "pca_features.csv",
        index=False,
    )


def save_loadings(
    loadings,
    output_folder,
):
    """
    Save PCA loading matrix.
    """

    output_folder = Path(output_folder)

    loadings.to_csv(
        output_folder / "loadings.csv"
    )


def save_variance_table(
    variance_table,
    output_folder,
):
    """
    Save explained variance table.
    """

    output_folder = Path(output_folder)

    variance_table.to_csv(
        output_folder / "explained_variance.csv",
        index=False,
    )


def save_summary(
    n_samples,
    n_features,
    retained_components,
    variance_threshold,
    explained_variance,
    output_folder,
):
    """
    Save a text summary of the PCA.
    """

    output_folder = Path(output_folder)

    summary = f"""
Principal Component Analysis Summary
------------------------------------

Samples               : {n_samples}

Original Features     : {n_features}

Retained Components   : {retained_components}

Variance Threshold    : {variance_threshold:.0%}

Variance Explained    : {explained_variance:.2%}
"""

    with open(
        output_folder / "pca_summary.txt",
        "w",
    ) as f:

        f.write(summary.strip())


def save_all_outputs(
    image_names,
    X_input,
    scores,
    loadings,
    variance_table,
    variance_threshold,
    output_folder,
):
    """
    Save every PCA output.
    """

    output_folder = Path(output_folder)

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_normalized_features(
        image_names,
        X_input,
        output_folder,
    )

    save_pca_scores(
        scores,
        output_folder,
    )

    save_loadings(
        loadings,
        output_folder,
    )

    save_variance_table(
        variance_table,
        output_folder,
    )

    retained = len(variance_table)

    explained = (
        variance_table[
            "Cumulative Variance"
        ].iloc[-1]
    )

    save_summary(
        n_samples=len(scores),
        n_features=X_input.shape[1],
        retained_components=retained,
        variance_threshold=variance_threshold,
        explained_variance=explained,
        output_folder=output_folder,
    )

    print(
        f"\nAll outputs written to:\n{output_folder}"
    )