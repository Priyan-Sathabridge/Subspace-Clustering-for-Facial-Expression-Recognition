"""
normalization.py

Loads a feature matrix from CSV and performs z-score normalization.

Expected CSV format:

Image,f1,f2,f3,...
img001,...
img002,...
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_features(csv_path):
    """
    Load feature matrix.

    Parameters
    ----------
    csv_path : str or Path

    Returns
    -------
    image_names : pandas.Series
        First column of the CSV.

    feature_names : list
        Names of feature columns.

    X : pandas.DataFrame
        Numeric feature matrix.
    """

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    if df.shape[1] < 2:
        raise ValueError(
            "CSV must contain an image column and at least one feature."
        )

    image_names = df.iloc[:, 0]

    X = df.iloc[:, 1:]

    feature_names = list(X.columns)

    return image_names, feature_names, X


def normalize_features(X):
    """
    Perform z-score normalization.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X_norm : pandas.DataFrame
    scaler : StandardScaler
    """

    scaler = StandardScaler()

    X_norm = scaler.fit_transform(X)

    X_norm = pd.DataFrame(
        X_norm,
        columns=X.columns,
        index=X.index
    )

    return X_norm, scaler


def save_normalized_features(
    image_names,
    X_norm,
    output_csv,
):
    """
    Save normalized feature matrix.
    """

    output_csv = Path(output_csv)

    df = pd.concat(
        [image_names.reset_index(drop=True),
         X_norm.reset_index(drop=True)],
        axis=1
    )

    df.to_csv(output_csv, index=False)

    print(f"Saved normalized features to:\n{output_csv}")