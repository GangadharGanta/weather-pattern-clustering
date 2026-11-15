"""Evaluation and Visualization Module: Generates PCA 2D scatter plots, Elbow curves, and radar charts."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_samples

from src.data_loader import CORE_METEOROLOGICAL_FEATURES
from src.utils import get_logger, load_artifact, load_json, plot_radar_chart

logger = get_logger("EvaluateClustering")

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]


def plot_pca_clusters(
    X_pca: np.ndarray,
    labels: np.ndarray,
    centroids_pca: np.ndarray,
    cluster_names: Dict[int, str],
    save_path: str = "models/evaluation_plots/pca_clusters_2d.png",
) -> plt.Figure:
    """Generate high-resolution 2D PCA cluster scatter plot with centroid markers."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    unique_labels = sorted(list(set(labels)))
    for idx, c_id in enumerate(unique_labels):
        mask = labels == c_id
        c_name = cluster_names.get(c_id, f"Cluster {c_id}")
        color = PALETTE[idx % len(PALETTE)]

        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f"Cluster {c_id}: {c_name}",
            alpha=0.60,
            edgecolors="none",
            s=45,
        )

    # Plot cluster centroids in PCA space
    ax.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c="black",
        s=220,
        marker="X",
        linewidths=2,
        edgecolors="white",
        label="Cluster Centroids",
        zorder=10,
    )

    ax.set_title("Weather Pattern Clusters in 2D Latent Space (PCA)", fontsize=14, weight="bold", pad=15)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11, weight="bold")
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11, weight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"PCA cluster plot saved to {save_path}")
    return fig


def plot_elbow_and_silhouette(
    tuning_metrics: List[Dict],
    save_path: str = "models/evaluation_plots/elbow_and_silhouette_tuning.png",
) -> plt.Figure:
    """Plot dual-axis Elbow Curve (WCSS) and Silhouette Score versus k."""
    ks = [m["k"] for m in tuning_metrics]
    wcss = [m["wcss"] for m in tuning_metrics]
    sils = [m["silhouette_score"] for m in tuning_metrics]

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_wcss = "#2980b9"
    color_sil = "#e67e22"

    ax1.set_xlabel("Number of Clusters (k)", fontsize=11, weight="bold")
    ax1.set_ylabel("Inertia / WCSS (Within-Cluster Sum of Squares)", color=color_wcss, fontsize=11, weight="bold")
    line1 = ax1.plot(ks, wcss, marker="o", color=color_wcss, linewidth=2.2, label="WCSS (Inertia)")
    ax1.tick_params(axis="y", labelcolor=color_wcss)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color=color_sil, fontsize=11, weight="bold")
    line2 = ax2.plot(ks, sils, marker="s", color=color_sil, linewidth=2.2, linestyle="--", label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_sil)

    # Highlight k=5 elbow/inflection
    if 5 in ks:
        idx_5 = ks.index(5)
        ax2.scatter([5], [sils[idx_5]], color="#c0392b", s=130, zorder=5, edgecolors="black")
        ax2.annotate(
            "Optimal k=5",
            xy=(5, sils[idx_5]),
            xytext=(5.3, sils[idx_5] + 0.03),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            fontweight="bold",
            color="#c0392b",
        )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", frameon=True)

    plt.title("Elbow Method & Silhouette Optimization for K-Means Clustering", fontsize=13, weight="bold", pad=12)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Elbow & Silhouette plot saved to {save_path}")
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    save_path: str = "models/evaluation_plots/feature_distributions_by_cluster.png",
) -> plt.Figure:
    """Generate multi-panel boxplots showing individual feature distributions per cluster."""
    features = CORE_METEOROLOGICAL_FEATURES
    n_features = len(features)

    fig, axes = plt.subplots(1, n_features, figsize=(18, 4.5), dpi=300)

    for i, feat in enumerate(features):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x=cluster_col,
            y=feat,
            ax=ax,
            palette=PALETTE,
            hue=cluster_col,
            legend=False,
            boxprops=dict(alpha=0.8),
        )
        ax.set_title(feat, fontsize=11, weight="bold")
        ax.set_xlabel("Cluster", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Meteorological Feature Distributions Across Weather Clusters", fontsize=14, weight="bold", y=1.03)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Feature distribution plots saved to {save_path}")
    return fig


def generate_all_evaluation_plots(
    model_path: str = "models/kmeans_champion.joblib",
    profiles_path: str = "models/cluster_profiles.json",
    metrics_path: str = "models/training_metrics.json",
    data_path: str = "data/processed/processed_weather_data.csv",
    plots_dir: str = "models/evaluation_plots",
) -> None:
    """Generate and export the complete suite of visual evaluation artifacts."""
    logger.info("Generating evaluation plots suite...")
    bundle = load_artifact(model_path)
    profiles = load_json(profiles_path)
    metrics = load_json(metrics_path)
    df = pd.read_csv(data_path)

    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]
    features = bundle["features"]

    X = df[features].values
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    labels = kmeans.labels_
    df["Cluster"] = labels

    cluster_names = {int(k): v["label"] for k, v in profiles.items()}
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # 1. 2D PCA Scatter plot
    plot_pca_clusters(
        X_pca=X_pca,
        labels=labels,
        centroids_pca=centroids_pca,
        cluster_names=cluster_names,
        save_path=f"{plots_dir}/pca_clusters_2d.png",
    )

    # 2. Elbow and Silhouette plot
    if "hyperparameter_search" in metrics:
        plot_elbow_and_silhouette(
            tuning_metrics=metrics["hyperparameter_search"],
            save_path=f"{plots_dir}/elbow_and_silhouette_tuning.png",
        )

    # 3. Feature Distributions Boxplot
    plot_feature_distributions(
        df=df,
        cluster_col="Cluster",
        save_path=f"{plots_dir}/feature_distributions_by_cluster.png",
    )

    # 4. Radar / Spider chart comparing normalized centroids
    categories = features
    cluster_means = {}
    for c_id, prof in profiles.items():
        # Min-max normalize centroid values for aesthetic radar comparison
        c_label = f"C{c_id}: {prof['climate_type']}"
        raw_vals = [prof["centroid"][feat] for feat in categories]
        cluster_means[c_label] = raw_vals

    # Standardize radar values between 0.1 and 1.0 for balanced polar geometry
    mins = [min(cluster_means[c][j] for c in cluster_means) for j in range(len(categories))]
    maxs = [max(cluster_means[c][j] for c in cluster_means) for j in range(len(categories))]

    norm_cluster_means = {}
    for c_label, vals in cluster_means.items():
        norm_vals = []
        for j, val in enumerate(vals):
            denom = maxs[j] - mins[j]
            norm_val = 0.5 if denom == 0 else (val - mins[j]) / denom
            norm_vals.append(round(norm_val, 3))
        norm_cluster_means[c_label] = norm_vals

    plot_radar_chart(
        categories=categories,
        cluster_means=norm_cluster_means,
        title="Comparative Meteorological Radar Profiles (Normalized)",
        save_path=f"{plots_dir}/radar_cluster_profiles.png",
    )

    logger.info("All evaluation plots generated successfully.")


if __name__ == "__main__":
    generate_all_evaluation_plots()
