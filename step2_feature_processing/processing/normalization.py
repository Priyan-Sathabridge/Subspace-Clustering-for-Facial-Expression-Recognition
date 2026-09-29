"""
normalization.py

Feature normalization methods for Step 2.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


EPS = 1e-8


# ==========================================================
# Base class
# ==========================================================

class BaseNormalizer(ABC):

    def __init__(self):

        self.statistics = {}

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)


class StandardScaler(BaseNormalizer):

    """
    Z-score normalization

    x = (x - mean) / std
    """

    def fit(self, X):

        self.mean = X.mean()

        self.std = X.std()

        self.statistics = {

            "mean": self.mean,

            "std": self.std

        }

    def transform(self, X):

        return (X - self.mean) / (self.std + EPS)


class MinMaxScaler(BaseNormalizer):

    """
    Min-Max normalization

    x = (x-min)/(max-min)
    """

    def fit(self, X):

        self.minimum = X.min()

        self.maximum = X.max()

        self.statistics = {

            "min": self.minimum,

            "max": self.maximum

        }

    def transform(self, X):

        return (

            X - self.minimum

        ) / (

            self.maximum -

            self.minimum +

            EPS

        )

class RobustScaler(BaseNormalizer):

    """
    Median/IQR scaling.

    Resistant to outliers.
    """

    def fit(self, X):

        self.median = X.median()

        self.q1 = X.quantile(0.25)

        self.q3 = X.quantile(0.75)

        self.iqr = self.q3 - self.q1

        self.statistics = {

            "median": self.median,

            "iqr": self.iqr

        }

    def transform(self, X):

        return (

            X -

            self.median

        ) / (

            self.iqr +

            EPS

        )

class Normalizer:

    METHODS = {

        "standard": StandardScaler,

        "minmax": MinMaxScaler,

        "robust": RobustScaler

    }

    def __init__(self, method="standard"):

        if method not in self.METHODS:

            raise ValueError(

                f"Unknown normalization method '{method}'."

            )

        self.scaler = self.METHODS[method]()

        self.method = method

    def process(self, df):

        df = df.copy()

        numeric = df.select_dtypes(include=np.number)

        scaled = self.scaler.fit_transform(numeric)

        df[scaled.columns] = scaled

        return df

    def get_statistics(self):

        return self.scaler.statistics