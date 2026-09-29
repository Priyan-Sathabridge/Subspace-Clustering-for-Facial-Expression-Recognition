"""
missing_values.py

Utilities for handling missing values.
"""

import pandas as pd


class MissingValueHandler:

    def __init__(self, method="drop"):

        valid = [

            "drop",
            "mean",
            "median",
            "zero"
        ]

        if method not in valid:

            raise ValueError(
                f"Unknown method: {method}"
            )

        self.method = method

    ########################################################

    def process(self, df):

        if self.method == "drop":

            return self._drop(df)

        elif self.method == "mean":

            return self._mean(df)

        elif self.method == "median":

            return self._median(df)

        elif self.method == "zero":

            return self._zero(df)

    ########################################################

    def _numeric_columns(self, df):

        return [

            c for c in df.columns

            if c != "Image"

        ]

    ########################################################

    def _drop(self, df):

        return df.dropna().reset_index(drop=True)

    ########################################################

    def _mean(self, df):

        df = df.copy()

        for col in self._numeric_columns(df):

            df[col] = df[col].fillna(
                df[col].mean()
            )

        return df

    ########################################################

    def _median(self, df):

        df = df.copy()

        for col in self._numeric_columns(df):

            df[col] = df[col].fillna(
                df[col].median()
            )

        return df

    ########################################################

    def _zero(self, df):

        df = df.copy()

        for col in self._numeric_columns(df):

            df[col] = df[col].fillna(0)

        return df