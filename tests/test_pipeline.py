"""Comprehensive automated test suite for the Weather Pattern Clustering pipeline."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ALL_NUMERIC_FEATURES,
    CORE_METEOROLOGICAL_FEATURES,
    clean_weather_data,
    generate_synthetic_weather_data,
    load_raw_data,
)
from src.feature_engineering import (
    build_pca_pipeline,
    build_scaling_pipeline,
    extract_features,
    fit_and_transform_pca,
)
from src.utils import (
    calculate_apparent_temperature,
    calculate_dew_point,
    compute_anomaly_score,
    load_artifact,
    load_json,
)


@pytest.fixture
def sample_raw_data():
    """Fixture providing synthetic daily weather DataFrame."""
    return generate_synthetic_weather_data(n_days=100, seed=123)


@pytest.fixture
def clean_data(sample_raw_data):
    """Fixture providing cleaned and imputed weather DataFrame."""
    return clean_weather_data(sample_raw_data)


def test_data_generation_shape_and_columns(sample_raw_data):
    """Verify raw data synthesis outputs expected dimensions and columns."""
    assert len(sample_raw_data) == 100
    for col in CORE_METEOROLOGICAL_FEATURES:
        assert col in sample_raw_data.columns
    assert "Date" in sample_raw_data.columns


def test_missing_value_imputation(sample_raw_data):
    """Verify that cleaning process eliminates all missing values."""
    cleaned = clean_weather_data(sample_raw_data)
    assert cleaned[CORE_METEOROLOGICAL_FEATURES].isnull().sum().sum() == 0


def test_physical_boundary_enforcement(sample_raw_data):
    """Ensure meteorological values fall strictly within physical Earth constraints."""
    cleaned = clean_weather_data(sample_raw_data)
    assert (cleaned["Humidity"] >= 1.0).all() and (cleaned["Humidity"] <= 100.0).all()
    assert (cleaned["Wind_Speed"] >= 0.0).all()
    assert (cleaned["Precipitation"] >= 0.0).all()
    assert (cleaned["Pressure"] >= 870.0).all() and (cleaned["Pressure"] <= 1085.0).all()


def test_derived_meteorological_formulas():
    """Verify Magnus-Tetens dew point and apparent temperature calculations."""
    temp, rh, wind = 30.0, 70.0, 15.0
    dew_pt = calculate_dew_point(temp, rh)
    assert 20.0 < dew_pt < 26.0

    app_temp = calculate_apparent_temperature(temp, rh, wind)
    assert app_temp > 25.0


def test_standard_scaler_pipeline(clean_data):
    """Assert feature scaling standardizes data to approximate zero mean and unit variance."""
    X = extract_features(clean_data, CORE_METEOROLOGICAL_FEATURES).values
    pipeline = build_scaling_pipeline()
    X_scaled = pipeline.fit_transform(X)

    assert X_scaled.shape == X.shape
    means = np.mean(X_scaled, axis=0)
    stds = np.std(X_scaled, axis=0)

    for m in means:
        assert abs(m) < 1e-7
    for s in stds:
        assert abs(s - 1.0) < 1e-7


def test_pca_transformation_shape(clean_data):
    """Verify PCA reduces 5D meteorological features into 2D latent coordinates."""
    X = extract_features(clean_data, CORE_METEOROLOGICAL_FEATURES).values
    X_pca, pca, scaler = fit_and_transform_pca(X, n_components=2)

    assert X_pca.shape == (len(clean_data), 2)
    assert len(pca.explained_variance_ratio_) == 2
    assert sum(pca.explained_variance_ratio_) > 0.40


def test_champion_model_prediction_and_bounds():
    """Ensure champion model loads, predicts valid cluster indices, and computes anomalies."""
    model_path = Path("models/kmeans_champion.joblib")
    if not model_path.exists():
        from src.train import train_and_select_champion
        train_and_select_champion()

    bundle = load_artifact(model_path)
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    k = bundle["optimal_k"]

    # Test single-point inference
    sample_input = np.array([[28.0, 80.0, 15.0, 1005.0, 10.0]])
    scaled_input = scaler.transform(sample_input)
    pred_cluster = int(kmeans.predict(scaled_input)[0])

    assert 0 <= pred_cluster < k
    centroid = kmeans.cluster_centers_[pred_cluster]
    anomaly_score = compute_anomaly_score(scaled_input[0], centroid)
    assert anomaly_score >= 0.0
    assert not math.isnan(anomaly_score)


def test_cluster_profiles_metadata():
    """Ensure cluster metadata JSON is well-formed with descriptive insights."""
    profiles_path = Path("models/cluster_profiles.json")
    if not profiles_path.exists():
        from src.train import train_and_select_champion
        train_and_select_champion()

    profiles = load_json(profiles_path)
    assert len(profiles) >= 3

    for c_id, prof in profiles.items():
        assert "label" in prof
        assert "centroid" in prof
        assert "agricultural_impact" in prof
        assert "energy_consumption" in prof
        assert "hazard_risk" in prof
