"""Unit tests for standalone prediction script and data validation module."""

from pathlib import Path
import pytest
from src.predict import predict_weather_regime
from src.validator import validate_dataframe_schema, validate_weather_vector


def test_cli_prediction_typical_scenario():
    """Verify that a standard spring day classifies correctly with low anomaly score."""
    res = predict_weather_regime(
        temperature=22.0,
        humidity=50.0,
        wind_speed=12.0,
        pressure=1015.0,
        precipitation=0.0,
    )
    assert "prediction" in res
    assert 0 <= res["prediction"]["cluster_id"] < 5
    assert res["prediction"]["anomaly_distance"] >= 0.0
    assert len(res["prediction"]["pca_2d_coordinates"]) == 2


def test_cli_prediction_extreme_anomaly():
    """Verify that an unprecedented physical reading gets flagged with elevated anomaly score."""
    res = predict_weather_regime(
        temperature=-35.0,
        humidity=10.0,
        wind_speed=95.0,
        pressure=965.0,
        precipitation=85.0,
    )
    assert res["prediction"]["anomaly_distance"] > 2.0
    assert res["prediction"]["anomaly_status"] in ["Moderate Deviation", "Extreme Anomaly"]


def test_validator_physical_bounds():
    """Ensure out-of-range sensor readings are detected by validator."""
    valid_data = {"Temperature": 25.0, "Humidity": 60.0, "Wind_Speed": 15.0, "Pressure": 1012.0, "Precipitation": 2.0}
    is_valid, violations = validate_weather_vector(valid_data)
    assert is_valid
    assert len(violations) == 0

    invalid_data = {"Temperature": 85.0, "Humidity": 140.0, "Wind_Speed": -10.0, "Pressure": 600.0, "Precipitation": -5.0}
    is_valid, violations = validate_weather_vector(invalid_data)
    assert not is_valid
    assert len(violations) >= 3
