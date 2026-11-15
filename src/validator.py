"""Data validation module ensuring physical meteorological boundary compliance and schema integrity."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

METEOROLOGICAL_PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Temperature": (-50.0, 60.0),       # Valid planetary surface temperatures in °C
    "Humidity": (0.0, 100.0),           # Relative humidity bounds in %
    "Wind_Speed": (0.0, 160.0),         # Maximum surface wind velocities in km/h
    "Pressure": (870.0, 1085.0),        # Typhoon low to Siberian high in hPa
    "Precipitation": (0.0, 500.0),      # Maximum 24-hr rainfall event in mm
}


def validate_weather_vector(features: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Validate that single-observation weather dictionary adheres to thermodynamic physical limits."""
    violations = []
    for feat_name, (min_val, max_val) in METEOROLOGICAL_PHYSICAL_BOUNDS.items():
        if feat_name in features:
            val = float(features[feat_name])
            if val < min_val or val > max_val:
                violations.append(
                    f"{feat_name} value {val} outside physical bounds [{min_val}, {max_val}]"
                )
    is_valid = len(violations) == 0
    return is_valid, violations


def validate_dataframe_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Assert required columns, zero NaN entries, and proper numerical datatypes."""
    required_cols = list(METEOROLOGICAL_PHYSICAL_BOUNDS.keys())
    missing = [c for c in required_cols if c not in df.columns]
    errors = []

    if missing:
        errors.append(f"Missing mandatory columns: {missing}")

    for c in required_cols:
        if c in df.columns:
            null_count = df[c].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{c}' contains {null_count} null entries.")
            if not np.issubdtype(df[c].dtype, np.number):
                errors.append(f"Column '{c}' is non-numeric (type: {df[c].dtype}).")

    is_valid = len(errors) == 0
    return is_valid, errors
