"""Utility functions for weather clustering pipeline, logging, and evaluation."""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_logger(name: str = "WeatherClustering", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def save_json(data: Union[Dict, List], filepath: Union[str, Path]) -> None:
    """Save dictionary or list to JSON file with indentation."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """Load and parse a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_artifact(obj: Any, filepath: Union[str, Path]) -> None:
    """Serialize an object using joblib."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(filepath: Union[str, Path]) -> Any:
    """Deserialize an object using joblib."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at: {path}")
    return joblib.load(path)


def calculate_dew_point(temp_c: float, humidity_pct: float) -> float:
    """
    Calculate Dew Point (°C) using the Magnus-Tetens approximation formula.
    Valid for temp between -45°C and 60°C and RH between 1% and 100%.
    """
    humidity_pct = max(0.1, min(100.0, float(humidity_pct)))
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(humidity_pct / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return round(dew_point, 2)


def calculate_apparent_temperature(
    temp_c: float, humidity_pct: float, wind_speed_kmh: float
) -> float:
    """
    Estimate Apparent Temperature (Australian Bureau of Meteorology formula).
    Combines ambient temp, water vapor pressure, and wind speed.
    """
    e = (humidity_pct / 100.0) * 6.105 * math.exp((17.27 * temp_c) / (237.7 + temp_c))
    v = wind_speed_kmh / 3.6  # convert km/h to m/s
    at = temp_c + 0.33 * e - 0.70 * v - 4.00
    return round(at, 2)


def compute_anomaly_score(
    point_scaled: np.ndarray, centroid_scaled: np.ndarray
) -> float:
    """
    Compute Euclidean distance between a scaled data point and its assigned cluster centroid.
    Larger distance represents higher deviation / anomaly severity.
    """
    dist = np.linalg.norm(point_scaled - centroid_scaled)
    return float(round(dist, 4))


def plot_radar_chart(
    categories: List[str],
    cluster_means: Dict[str, List[float]],
    title: str = "Cluster Meteorological Profiles (Normalized)",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Generate a radar/spider chart comparing standardized feature means across clusters.
    """
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for i, (label, values) in enumerate(cluster_means.items()):
        vals = values + values[:1]  # Loop closed
        color = colors[i % len(colors)]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10, weight="bold")
    ax.set_title(title, size=14, weight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
