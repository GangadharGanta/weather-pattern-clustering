"""CLI prediction script for real-time weather pattern categorization and anomaly scoring."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import compute_anomaly_score, load_artifact, load_json


def predict_weather_regime(
    temperature: float,
    humidity: float,
    wind_speed: float,
    pressure: float,
    precipitation: float,
    model_path: str = "models/kmeans_champion.joblib",
    profiles_path: str = "models/cluster_profiles.json",
) -> dict:
    """Classify a weather vector and return cluster metadata with anomaly severity."""
    m_path = project_root / model_path
    p_path = project_root / profiles_path

    bundle = load_artifact(m_path)
    profiles = load_json(p_path)

    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]

    input_raw = np.array([[temperature, humidity, wind_speed, pressure, precipitation]])
    input_scaled = scaler.transform(input_raw)

    cluster_id = int(kmeans.predict(input_scaled)[0])
    centroid_scaled = kmeans.cluster_centers_[cluster_id]
    anomaly_distance = compute_anomaly_score(input_scaled[0], centroid_scaled)

    pca_coords = pca.transform(input_scaled)[0].tolist()
    info = profiles.get(str(cluster_id), {})

    status = (
        "Typical Observation"
        if anomaly_distance < 2.0
        else ("Moderate Deviation" if anomaly_distance < 3.5 else "Extreme Anomaly")
    )

    result = {
        "input": {
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure,
            "precipitation_mm": precipitation,
        },
        "prediction": {
            "cluster_id": cluster_id,
            "regime_name": info.get("label", f"Cluster {cluster_id}"),
            "climate_type": info.get("climate_type", "Localized Regime"),
            "anomaly_distance": round(anomaly_distance, 4),
            "anomaly_status": status,
            "pca_2d_coordinates": [round(c, 4) for c in pca_coords],
        },
        "advisory": {
            "agricultural_impact": info.get("agricultural_impact", "Standard monitoring."),
            "energy_consumption": info.get("energy_consumption", "Nominal grid demand."),
            "hazard_risk": info.get("hazard_risk", "Low"),
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify daily weather parameters into a climate cluster.")
    parser.add_argument("--temp", type=float, required=True, help="Ambient temperature in °C")
    parser.add_argument("--humidity", type=float, required=True, help="Relative humidity (1 to 100%)")
    parser.add_argument("--wind", type=float, default=15.0, help="Wind speed in km/h")
    parser.add_argument("--pressure", type=float, default=1013.25, help="Atmospheric pressure in hPa")
    parser.add_argument("--rain", type=float, default=0.0, help="Precipitation in mm")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()
    res = predict_weather_regime(
        temperature=args.temp,
        humidity=args.humidity,
        wind_speed=args.wind,
        pressure=args.pressure,
        precipitation=args.rain,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print("\n=== Weather Pattern Clustering Inference Result ===")
        print(f"Predicted Regime : Cluster {res['prediction']['cluster_id']} - {res['prediction']['regime_name']}")
        print(f"Climate Type     : {res['prediction']['climate_type']}")
        print(f"Anomaly Score    : {res['prediction']['anomaly_distance']} ({res['prediction']['anomaly_status']})")
        print(f"PCA 2D Embedding : PC1={res['prediction']['pca_2d_coordinates'][0]}, PC2={res['prediction']['pca_2d_coordinates'][1]}")
        print(f"Hazard Warning   : {res['advisory']['hazard_risk']}")
        print("===================================================\n")


if __name__ == "__main__":
    main()
