"""Feature Engineering, Scaling Pipelines, and Dimensionality Reduction (PCA)."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils import get_logger

logger = get_logger("FeatureEngineering")

DEFAULT_FEATURES = [
    "Temperature",
    "Humidity",
    "Wind_Speed",
    "Pressure",
    "Precipitation",
]


class CyclicalDateEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes datetime features (month, day of year) into sin/cos cyclical representations
    to preserve seasonal continuity (e.g., December is adjacent to January).
    """

    def __init__(self, date_column: str = "Date"):
        self.date_column = date_column

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if self.date_column in X_out.columns:
            dates = pd.to_datetime(X_out[self.date_column])
            day_of_year = dates.dt.dayofyear
            X_out["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / 365.25)
            X_out["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / 365.25)
        return X_out


def build_scaling_pipeline(features: Optional[List[str]] = None) -> Pipeline:
    """
    Construct the baseline feature scaling pipeline with StandardScaler.
    Distance-based unsupervised clustering (K-Means, Agglomerative, DBSCAN)
    fundamentally requires feature standardization to mean=0, variance=1.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


def build_pca_pipeline(
    n_components: Union[int, float] = 2,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct a full transformation pipeline: StandardScaler followed by PCA.
    Used for 2D latent manifold projection, visual inspection, and noise reduction.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
        ]
    )
    return pipeline


def extract_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract and validate the required numerical feature columns."""
    target_cols = features if features is not None else DEFAULT_FEATURES
    missing_cols = [c for c in target_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Features missing from input DataFrame: {missing_cols}")
    return df[target_cols].copy()


def fit_and_transform_pca(
    X: Union[pd.DataFrame, np.ndarray],
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, PCA, StandardScaler]:
    """
    Fit StandardScaler and PCA, returning transformed 2D coordinates,
    the fitted PCA model, and the fitted StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    explained_var = pca.explained_variance_ratio_
    logger.info(
        f"PCA fitted ({n_components} components). "
        f"Explained variance ratio: {explained_var.round(4)} "
        f"(Total: {explained_var.sum():.2%})"
    )
    return X_pca, pca, scaler
