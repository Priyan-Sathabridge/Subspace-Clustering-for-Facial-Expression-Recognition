"""
validator.py

Validates the feature matrix before preprocessing.
"""

from pathlib import Path
import pandas as pd
import numpy as np


class FeatureValidator:

    def __init__(self):
        self.report = {}

    def validate(self, df):

        self.report = {}

        self._check_empty(df)
        self._check_duplicate_images(df)
        self._check_duplicate_columns(df)
        self._check_non_numeric(df)
        self._check_missing(df)
        self._check_infinite(df)
        self._check_constant_features(df)

        return self.report

    ########################################################

    def _check_empty(self, df):

        if df.empty:
            raise ValueError("Feature matrix is empty.")

        self.report["Rows"] = len(df)
        self.report["Columns"] = len(df.columns)

    ########################################################

    def _check_duplicate_images(self, df):

        if "Image" not in df.columns:
            return

        duplicates = df["Image"].duplicated().sum()

        self.report["Duplicate Images"] = int(duplicates)

    ########################################################

    def _check_duplicate_columns(self, df):

        duplicates = int(df.columns.duplicated().sum())

        self.report["Duplicate Columns"] = duplicates

    ########################################################

    def _check_non_numeric(self, df):

        numeric = df.select_dtypes(include=np.number).columns

        non_numeric = [

            c for c in df.columns

            if c not in numeric and c != "Image"

        ]

        self.report["Non Numeric Columns"] = non_numeric

    ########################################################

    def _check_missing(self, df):

        missing = df.isna().sum()

        self.report["Missing Values"] = {

            k: int(v)

            for k, v in missing.items()

            if v > 0

        }

    ########################################################

    def _check_infinite(self, df):

        numeric = df.select_dtypes(include=np.number)

        inf = np.isinf(numeric).sum()

        self.report["Infinite Values"] = {

            k: int(v)

            for k, v in inf.items()

            if v > 0

        }

    ########################################################

    def _check_constant_features(self, df):

        numeric = df.select_dtypes(include=np.number)

        constant = []

        for col in numeric.columns:

            if numeric[col].nunique() <= 1:

                constant.append(col)

        self.report["Constant Features"] = constant

    ########################################################

    def print_report(self):

        print("\nValidation Report")
        print("=" * 50)

        for key, value in self.report.items():

            print(f"{key}: {value}")

    ########################################################

    def save_report(self, output_file):

        output_file = Path(output_file)

        with open(output_file, "w") as f:

            f.write("Validation Report\n")
            f.write("=" * 50 + "\n\n")

            for key, value in self.report.items():

                f.write(f"{key}: {value}\n")