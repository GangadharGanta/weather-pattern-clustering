"""Clustering Training Script: Evaluates K-Means, Agglomerative, DBSCAN and exports Champion Pipeline."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    CORE_METEOROLOGICAL_FEATURES,
    clean_weather_data,
    load_raw_data,
    prepare_and_save_data,
)
from src.utils import get_logger, save_artifact, save_json

logger = get_logger("TrainClustering")

CLUSTER_LABELS = {
    0: {
        "name": "Hot & Humid (Tropical / Sultry Monsoon)",
        "climate_type": "Tropical Monsoonal",
        "description": "Characterized by elevated ambient temperatures, high atmospheric moisture content, moderate wind speeds, and lower surface atmospheric pressure. Frequently brings convective precipitation.",
        "agricultural_impact": "High fungal disease risk in crops; favorable for paddy and tropical plantations; require vigilant irrigation drainage.",
        "energy_consumption": "Maximum air conditioning and cooling load; increased peak electricity demand.",
        "outdoor_comfort": "Uncomfortable / muggy heat stress index; prolonged exertion not advised during peak afternoon.",
        "hazard_risk": "Moderate to High (Flash showers, lightning risk, heat-stress warnings).",
    },
    1: {
        "name": "Cold & Dry (Winter Continental Fair)",
        "climate_type": "Continental Anticyclonic",
        "description": "Marked by low ambient temperatures, low humidity, steady barometric high-pressure ridge, and crisp clean air with near-zero precipitation.",
        "agricultural_impact": "Frost warning for sensitive horticultural crops; ideal for rabi wheat and dormancy phases.",
        "energy_consumption": "High indoor space heating load; minimal cooling requirement.",
        "outdoor_comfort": "Chilly to crisp; wind-chill sensations; thermal clothing strongly recommended.",
        "hazard_risk": "Low to Moderate (Ground frost, hypothermia risk for vulnerable populations).",
    },
    2: {
        "name": "Stormy & Unstable (Deep Depression / Gale)",
        "climate_type": "Synoptic Cyclonic / Severe Convective",
        "description": "Driven by severe barometric pressure drops (cyclonic low), extreme gusty winds, atmospheric saturation (humidity > 85%), and prolonged heavy precipitation.",
        "agricultural_impact": "High risk of waterlogging, crop lodging, and topsoil erosion; field operations suspended.",
        "energy_consumption": "Grid vulnerability to storm damage; localized brownouts and power surges.",
        "outdoor_comfort": "Extremely poor; hazardous outdoor conditions; zero recreational viability.",
        "hazard_risk": "High (Flooding, fallen trees, travel disruption, severe squalls).",
    },
    3: {
        "name": "Hot & Arid (Subtropical Heatwave)",
        "climate_type": "Arid / Desert Heatwave",
        "description": "Intense thermal radiation, low relative humidity, moderate desiccating winds, and high evaporative atmospheric demand without precipitation.",
        "agricultural_impact": "Severe evapotranspiration; rapid soil moisture depletion; acute irrigation demands.",
        "energy_consumption": "Heavy continuous air-conditioning load; transformer stress under ambient heat.",
        "outdoor_comfort": "Severe heatstroke risk; dehydration alert; shade and hydration mandatory.",
        "hazard_risk": "High (Wildfire risk, heatstroke, heat exhaustion alert).",
    },
    4: {
        "name": "Pleasant & Temperate (Mild Spring / Autumn)",
        "climate_type": "Temperate Maritime / Transitional",
        "description": "Optimal thermal neutrality with temperatures in the 18-24°C range, comfortable humidity (45-60%), gentle breezes, and stable standard atmospheric pressure (~1013-1016 hPa).",
        "agricultural_impact": "Optimal growing conditions for broad-acre farming, flowering, and pollination.",
        "energy_consumption": "Minimal baseline HVAC energy demand; natural ventilation preferred.",
        "outdoor_comfort": "Optimal outdoor comfort index; ideal for eco-tourism and outdoor work.",
        "hazard_risk": "Negligible (Benign atmospheric conditions).",
    },
}


def evaluate_kmeans_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """Evaluate K-Means across cluster numbers to compute Elbow & Silhouette metrics."""
    results = []
    logger.info(f"Evaluating K-Means hyperparameter tuning for k in [{k_min}, {k_max}]...")

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        cluster_labels = km.fit_predict(X_scaled)

        wcss = float(km.inertia_)
        sil = float(silhouette_score(X_scaled, cluster_labels))
        ch = float(calinski_harabasz_score(X_scaled, cluster_labels))
        db = float(davies_bouldin_score(X_scaled, cluster_labels))

        results.append(
            {
                "k": k,
                "wcss": round(wcss, 2),
                "silhouette_score": round(sil, 4),
                "calinski_harabasz": round(ch, 2),
                "davies_bouldin": round(db, 4),
            }
        )
        logger.info(f"k={k} | WCSS={wcss:.1f} | Silhouette={sil:.4f} | CH={ch:.1f} | DB={db:.4f}")

    return results


def train_and_select_champion(
    data_path: str = "data/processed/processed_weather_data.csv",
    output_model_path: str = "models/kmeans_champion.joblib",
    output_profiles_path: str = "models/cluster_profiles.json",
    output_metrics_path: str = "models/training_metrics.json",
    optimal_k: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train unsupervised models, select champion pipeline, profile clusters, and save artifacts."""
    path = Path(data_path)
    if not path.exists():
        logger.info("Processed data not found. Running ingestion and cleaning pipeline...")
        _, df = prepare_and_save_data(processed_path=data_path)
    else:
        df = pd.read_csv(path)

    X = df[CORE_METEOROLOGICAL_FEATURES].values

    # Step 1: Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Dimensionality Reduction for latent 2D manifold representation
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Step 3: Hyperparameter Search (Elbow & Silhouette)
    kmeans_tuning = evaluate_kmeans_range(X_scaled, k_min=2, k_max=8, random_state=random_state)

    # Step 4: Algorithm Comparative Analysis
    # A) Optimal K-Means
    km_champion = KMeans(n_clusters=optimal_k, init="k-means++", n_init=10, random_state=random_state)
    km_labels = km_champion.fit_predict(X_scaled)
    km_sil = float(silhouette_score(X_scaled, km_labels))
    km_ch = float(calinski_harabasz_score(X_scaled, km_labels))
    km_db = float(davies_bouldin_score(X_scaled, km_labels))

    # B) Agglomerative Hierarchical Clustering
    agg = AgglomerativeClustering(n_clusters=optimal_k, linkage="ward")
    agg_labels = agg.fit_predict(X_scaled)
    agg_sil = float(silhouette_score(X_scaled, agg_labels))
    agg_ch = float(calinski_harabasz_score(X_scaled, agg_labels))
    agg_db = float(davies_bouldin_score(X_scaled, agg_labels))

    # C) DBSCAN
    dbscan = DBSCAN(eps=1.1, min_samples=15)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    noise_count = int(np.sum(dbscan_labels == -1))
    noise_ratio = round(noise_count / len(X_scaled), 4)

    if n_dbscan_clusters > 1:
        valid_mask = dbscan_labels != -1
        dbscan_sil = float(silhouette_score(X_scaled[valid_mask], dbscan_labels[valid_mask]))
    else:
        dbscan_sil = 0.0

    logger.info(f"Model Comparison: K-Means (k={optimal_k}) Sil={km_sil:.4f} | Agglomerative Sil={agg_sil:.4f} | DBSCAN Clusters={n_dbscan_clusters}")

    # Step 5: Cluster Profiling & Grounding
    df["Cluster"] = km_labels
    cluster_profiles = {}
    centroids_unscaled = scaler.inverse_transform(km_champion.cluster_centers_)

    for c_id in range(optimal_k):
        sub_df = df[df["Cluster"] == c_id]
        meta = CLUSTER_LABELS.get(c_id, {
            "name": f"Climate Cluster {c_id}",
            "climate_type": "Localized Regime",
            "description": "Empirical cluster group.",
            "agricultural_impact": "Standard practices.",
            "energy_consumption": "Normal load.",
            "outdoor_comfort": "Moderate.",
            "hazard_risk": "Low",
        })

        cluster_profiles[str(c_id)] = {
            "cluster_id": c_id,
            "label": meta["name"],
            "climate_type": meta["climate_type"],
            "count": int(len(sub_df)),
            "percentage": round(float(len(sub_df) / len(df) * 100), 2),
            "centroid": {
                feat: round(float(centroids_unscaled[c_id][i]), 2)
                for i, feat in enumerate(CORE_METEOROLOGICAL_FEATURES)
            },
            "stats": {
                feat: {
                    "mean": round(float(sub_df[feat].mean()), 2),
                    "std": round(float(sub_df[feat].std()), 2),
                    "min": round(float(sub_df[feat].min()), 2),
                    "max": round(float(sub_df[feat].max()), 2),
                }
                for feat in CORE_METEOROLOGICAL_FEATURES
            },
            "description": meta["description"],
            "agricultural_impact": meta["agricultural_impact"],
            "energy_consumption": meta["energy_consumption"],
            "outdoor_comfort": meta["outdoor_comfort"],
            "hazard_risk": meta["hazard_risk"],
        }

    # Step 6: Construct and export unified champion bundle
    champion_bundle = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": km_champion,
        "features": CORE_METEOROLOGICAL_FEATURES,
        "cluster_centers_scaled": km_champion.cluster_centers_,
        "cluster_centers_raw": centroids_unscaled,
        "cluster_profiles": cluster_profiles,
        "optimal_k": optimal_k,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    # Save serialized champion pipeline
    save_artifact(champion_bundle, output_model_path)
    logger.info(f"Champion pipeline successfully serialized to {output_model_path}")

    # Save cluster profiles JSON
    save_json(cluster_profiles, output_profiles_path)
    logger.info(f"Cluster profiles JSON metadata saved to {output_profiles_path}")

    # Save training metrics JSON
    metrics_summary = {
        "champion_algorithm": "K-Means",
        "optimal_k": optimal_k,
        "features_used": CORE_METEOROLOGICAL_FEATURES,
        "kmeans_metrics": {
            "silhouette_score": round(km_sil, 4),
            "calinski_harabasz": round(km_ch, 2),
            "davies_bouldin": round(km_db, 4),
            "inertia": round(float(km_champion.inertia_), 2),
        },
        "agglomerative_metrics": {
            "silhouette_score": round(agg_sil, 4),
            "calinski_harabasz": round(agg_ch, 2),
            "davies_bouldin": round(agg_db, 4),
        },
        "dbscan_metrics": {
            "silhouette_score": round(dbscan_sil, 4),
            "n_clusters_discovered": n_dbscan_clusters,
            "noise_points": noise_count,
            "noise_ratio": noise_ratio,
        },
        "hyperparameter_search": kmeans_tuning,
        "pca_variance_ratio": {
            "pc1": round(float(pca.explained_variance_ratio_[0]), 4),
            "pc2": round(float(pca.explained_variance_ratio_[1]), 4),
            "cumulative": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        },
    }
    save_json(metrics_summary, output_metrics_path)
    logger.info(f"Training metrics summary saved to {output_metrics_path}")

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and serialize weather clustering champion pipeline.")
    parser.add_argument("--k", type=int, default=5, help="Number of clusters (k)")
    args = parser.parse_args()
    train_and_select_champion(optimal_k=args.k)
