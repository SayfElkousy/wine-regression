from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WineFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create physically interpretable wine chemistry features."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enabled:
            return X

        eps = 1e-6
        X["bound_sulfur_dioxide"] = X["total sulfur dioxide"] - X["free sulfur dioxide"]
        X["free_to_total_so2_ratio"] = X["free sulfur dioxide"] / (X["total sulfur dioxide"] + eps)
        X["fixed_to_volatile_acidity"] = X["fixed acidity"] / (X["volatile acidity"] + eps)
        X["citric_to_fixed_acidity"] = X["citric acid"] / (X["fixed acidity"] + eps)
        X["sugar_density_ratio"] = X["residual sugar"] / (X["density"] + eps)
        X["alcohol_density_ratio"] = X["alcohol"] / (X["density"] + eps)
        X["sulphates_chlorides_ratio"] = X["sulphates"] / (X["chlorides"] + eps)
        X["alcohol_x_sulphates"] = X["alcohol"] * X["sulphates"]
        X["alcohol_x_volatile_acidity"] = X["alcohol"] * X["volatile acidity"]
        X["acidity_x_pH"] = (
            X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]
        ) * X["pH"]
        X["total_acidity_proxy"] = X["fixed acidity"] + X["volatile acidity"] + X["citric acid"]

        for col in [
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "sulphates",
        ]:
            X[f"log_{col.replace(' ', '_')}"] = np.log1p(np.maximum(X[col], 0))

        return X


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns using train-fold quantiles only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y=None):
        frame = pd.DataFrame(X).copy()
        self.columns_ = list(frame.columns)
        self.lower_bounds_ = frame.quantile(self.lower, numeric_only=True)
        self.upper_bounds_ = frame.quantile(self.upper, numeric_only=True)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        frame.columns = self.columns_
        return frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
