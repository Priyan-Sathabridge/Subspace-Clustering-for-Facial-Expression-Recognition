"""
weighting.py

Group-based feature weighting.
"""

import numpy as np
import pandas as pd

def build_feature_groups(df):
    """
    Automatically group features based on
    their column names.
    """

    groups = {
        "Eyes": [],
        "Mouth": [],
        "Eyebrows": []
    }

    for col in df.columns:

        if col == "Image":
            continue

        # Eyes
        if (
            col.startswith("Left Eye")
            or col.startswith("Right Eye")
            or col.startswith("Eye ")
            or col.startswith("Inter-Eye")
            or "EAR" in col
        ):
            groups["Eyes"].append(col)

        # Mouth
        elif (
            col.startswith("Mouth")
            or col.startswith("Upper Lip")
            or col == "MAR"
        ):
            groups["Mouth"].append(col)

        # Eyebrows
        elif (
            col.startswith("Left Brow")
            or col.startswith("Right Brow")
            or col.startswith("Eyebrow")
            or col.startswith("Inner Brow")
            or col.startswith("Brow Centre")
        ):
            groups["Eyebrows"].append(col)

    return groups


class FeatureWeighter:

    """
    Weight features according to
    the number of features in each group.
    """

    def __init__(self,
                 feature_groups,
                 method="sqrt"):

        valid = [

            "equal",
            "inverse",
            "sqrt"
        ]

        if method not in valid:

            raise ValueError(
                f"Unknown weighting method '{method}'"
            )

        self.feature_groups = feature_groups
        self.method = method

        self.weights = {}

    #####################################################

    def process(self, df):

        df = df.copy()

        numeric_columns = [

            c for c in df.columns

            if c != "Image"
        ]

        self.weights = {}

        for group, columns in self.feature_groups.items():

            columns = [

                c for c in columns

                if c in numeric_columns
            ]

            if len(columns) == 0:

                continue

            n = len(columns)

            if self.method == "equal":

                weight = 1.0

            elif self.method == "inverse":

                weight = 1.0 / n

            elif self.method == "sqrt":

                weight = 1.0 / np.sqrt(n)

            for col in columns:

                df[col] *= weight

                self.weights[col] = {

                    "Group": group,
                    "Weight": weight
                }

        return df

    #####################################################

    def save_weights(self, filename):

        weight_df = pd.DataFrame(

            self.weights

        ).T

        weight_df.index.name = "Feature"

        weight_df.to_csv(filename)

