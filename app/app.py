"""Streamlit Interactive Web Application for Weather Pattern Clustering and Climate Insights."""

import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of how Streamlit is launched
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Configure page configuration
st.set_page_config(
    page_title="Weather Pattern Clustering | INT396",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-card {
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .info-card {
        background-color: #f1f5f9;
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
BG_COLORS = ["#b91c1c", "#1d4ed8", "#15803d", "#b45309", "#7e22ce"]

PRESETS = {
    "Select a Scenario Preset...": None,
    "🌧️ Monsoon / Heavy Rain": {"temp": 27.5, "humidity": 88.0, "wind": 22.0, "pressure": 1002.0, "precip": 38.0},
    "❄️ Winter Continental Frost": {"temp": 4.5, "humidity": 32.0, "wind": 11.0, "pressure": 1026.0, "precip": 0.0},
    "☀️ Desert Summer Heatwave": {"temp": 42.0, "humidity": 18.0, "wind": 26.0, "pressure": 1008.0, "precip": 0.0},
    "🌪️ Severe Depression / Gale": {"temp": 19.0, "humidity": 94.0, "wind": 48.0, "pressure": 988.0, "precip": 62.0},
    "🌸 Pleasant Spring Day": {"temp": 22.0, "humidity": 52.0, "wind": 10.0, "pressure": 1015.0, "precip": 0.5},
}


@st.cache_resource
def load_model_and_metadata():
    """Load the champion model artifacts and metadata."""
    import joblib
    from src.train import train_and_select_champion
    from src.utils import load_artifact, load_json

    model_path = project_root / "models" / "kmeans_champion.joblib"
    profiles_path = project_root / "models" / "cluster_profiles.json"
    metrics_path = project_root / "models" / "training_metrics.json"
    data_path = project_root / "data" / "processed" / "processed_weather_data.csv"

    if not model_path.exists() or not profiles_path.exists():
        train_and_select_champion()

    bundle = load_artifact(model_path)
    profiles = load_json(profiles_path)
    metrics = load_json(metrics_path)
    df = pd.read_csv(data_path) if data_path.exists() else None

    return bundle, profiles, metrics, df


def main():
    bundle, profiles, metrics, df = load_model_and_metadata()
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    kmeans = bundle["kmeans"]
    features = bundle["features"]

    # Header section
    st.markdown('<div class="main-title">🌤️ Weather Pattern Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title"><b>Course: INT396 (Unsupervised ML)</b> | <b>Problem ID: 31</b> | '
        '<b>Student: Ganta Gangadhar (Reg No: 12523695)</b> | '
        '<a href="https://github.com/GangadharGanta/weather-pattern-clustering" target="_blank">GitHub Repository</a> | '
        '<a href="https://weather-pattern-clustering.streamlit.app" target="_blank">Live Deployment</a></div>',
        unsafe_allow_html=True,
    )

    # Sidebar inputs
    st.sidebar.header("🕹️ Daily Climate Input Controls")
    preset_choice = st.sidebar.selectbox("⚡ Quick Climate Presets", list(PRESETS.keys()))

    preset_vals = PRESETS.get(preset_choice)

    temp_default = preset_vals["temp"] if preset_vals else 28.0
    hum_default = preset_vals["humidity"] if preset_vals else 75.0
    wind_default = preset_vals["wind"] if preset_vals else 14.0
    press_default = preset_vals["pressure"] if preset_vals else 1008.0
    precip_default = preset_vals["precip"] if preset_vals else 8.0

    temp_in = st.sidebar.slider("🌡️ Temperature (°C)", min_value=-20.0, max_value=52.0, value=float(temp_default), step=0.5)
    humidity_in = st.sidebar.slider("💧 Relative Humidity (%)", min_value=1.0, max_value=100.0, value=float(hum_default), step=1.0)
    wind_in = st.sidebar.slider("💨 Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=float(wind_default), step=1.0)
    pressure_in = st.sidebar.slider("⏱️ Atmospheric Pressure (hPa)", min_value=960.0, max_value=1050.0, value=float(press_default), step=1.0)
    precip_in = st.sidebar.slider("🌧️ Precipitation (mm/day)", min_value=0.0, max_value=120.0, value=float(precip_default), step=0.5)

    input_vector = np.array([[temp_in, humidity_in, wind_in, pressure_in, precip_in]])

    # Inference pipeline
    input_scaled = scaler.transform(input_vector)
    cluster_pred = int(kmeans.predict(input_scaled)[0])
    input_pca = pca.transform(input_scaled)[0]

    # Compute Euclidean distance to cluster centroid (anomaly measure)
    centroid_scaled = kmeans.cluster_centers_[cluster_pred]
    anomaly_dist = float(np.linalg.norm(input_scaled - centroid_scaled))

    cluster_info = profiles.get(str(cluster_pred), {})
    cluster_name = cluster_info.get("label", f"Cluster {cluster_pred}")
    climate_type = cluster_info.get("climate_type", "Localized Regime")
    bg_color = BG_COLORS[cluster_pred % len(BG_COLORS)]

    # Anomaly status determination
    anomaly_status = "Typical Observation" if anomaly_dist < 2.0 else ("Moderate Deviation" if anomaly_dist < 3.5 else "Extreme Anomaly")
    anomaly_color = "#10b981" if anomaly_dist < 2.0 else ("#f59e0b" if anomaly_dist < 3.5 else "#ef4444")

    # Main inference presentation
    st.markdown(
        f"""
        <div class="badge-card" style="background-color: {bg_color};">
            <h2 style="margin:0; font-size:1.7rem; color:white;">📍 Assigned Regime: Cluster {cluster_pred}</h2>
            <h3 style="margin:0.2rem 0 0.8rem 0; font-weight:400; font-size:1.25rem; color:#f8fafc;">{cluster_name}</h3>
            <p style="margin:0; font-size:0.95rem; opacity:0.92;"><b>Synoptic Classification:</b> {climate_type}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key metric cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value">{cluster_pred}</div>
            <div class="metric-label">Predicted Cluster</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="color:{anomaly_color};">{anomaly_dist:.2f}</div>
            <div class="metric-label">Centroid Distance (Anomaly)</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value" style="font-size:1.15rem; color:{anomaly_color}; padding-top:0.35rem;">{anomaly_status}</div>
            <div class="metric-label">Observation Status</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-box"><div class="metric-value">{cluster_info.get('percentage', 20.0)}%</div>
            <div class="metric-label">Historical Frequency</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    # Detailed domain insights
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📖 Meteorological & Agricultural Insights")
        st.markdown(
            f"""
            <div class="info-card">
                <b>Synoptic Description:</b><br>{cluster_info.get('description', 'N/A')}
            </div>
            <div class="info-card" style="border-left-color: #10b981;">
                <b>🌾 Agricultural & Agronomic Impact:</b><br>{cluster_info.get('agricultural_impact', 'N/A')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### ⚡ Infrastructure, Energy & Public Safety")
        st.markdown(
            f"""
            <div class="info-card" style="border-left-color: #f59e0b;">
                <b>⚡ Energy & Grid Load:</b><br>{cluster_info.get('energy_consumption', 'N/A')}
            </div>
            <div class="info-card" style="border-left-color: #ef4444;">
                <b>⚠️ Atmospheric Hazard & Comfort Advisory:</b><br>
                <b>Comfort:</b> {cluster_info.get('outdoor_comfort', 'N/A')}<br>
                <b>Hazard Rating:</b> {cluster_info.get('hazard_risk', 'Low')}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Interactive visual tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📍 2D Latent PCA Map",
        "🕸️ Meteorological Radar Profile",
        "📊 Historical Cluster Directory",
        "🧠 Model Architecture & Metrics",
    ])

    with tab1:
        st.subheader("2D Principal Component Projection with Active Input")
        st.write(
            "Visualizing multidimensional meteorological space projected onto 2 principal components. "
            "The **★ Golden Star** indicates your current slider input relative to historical clusters."
        )

        if df is not None:
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

            X_hist = df[features].values
            X_hist_scaled = scaler.transform(X_hist)
            X_hist_pca = pca.transform(X_hist_scaled)
            hist_labels = kmeans.predict(X_hist_scaled)

            for c_id in range(bundle["optimal_k"]):
                mask = hist_labels == c_id
                c_lbl = profiles.get(str(c_id), {}).get("label", f"Cluster {c_id}")
                ax.scatter(
                    X_hist_pca[mask, 0],
                    X_hist_pca[mask, 1],
                    c=PALETTE[c_id % len(PALETTE)],
                    label=f"C{c_id}: {c_lbl.split('(')[0]}",
                    alpha=0.45,
                    s=32,
                    edgecolors="none",
                )

            # Cluster Centroids in PCA space
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            ax.scatter(
                centroids_pca[:, 0],
                centroids_pca[:, 1],
                c="black",
                marker="X",
                s=180,
                edgecolors="white",
                label="Centroids",
                zorder=10,
            )

            # User point
            ax.scatter(
                [input_pca[0]],
                [input_pca[1]],
                c="#ffd700",
                marker="*",
                s=400,
                edgecolors="#b91c1c",
                linewidths=2.2,
                label=f"Current Input (PC1={input_pca[0]:.2f}, PC2={input_pca[1]:.2f})",
                zorder=20,
            )

            ax.set_xlabel(f"PC1 ({bundle['explained_variance_ratio'][0]:.1%} Variance)", weight="bold")
            ax.set_ylabel(f"PC2 ({bundle['explained_variance_ratio'][1]:.1%} Variance)", weight="bold")
            ax.legend(loc="upper right", frameon=True, fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.subheader("Normalized Radar Profile (Input vs Centroid)")
        from src.utils import plot_radar_chart

        categories = features
        raw_centroid = [cluster_info["centroid"][f] for f in categories]
        raw_user = [temp_in, humidity_in, wind_in, pressure_in, precip_in]

        # Min-max normalization based on realistic extremes for radar plotting
        mins = [-10.0, 5.0, 0.0, 970.0, 0.0]
        maxs = [45.0, 100.0, 60.0, 1040.0, 70.0]

        def normalize_vec(vec):
            return [max(0.05, min(1.0, (val - mins[i]) / (maxs[i] - mins[i]))) for i, val in enumerate(vec)]

        radar_data = {
            f"Cluster {cluster_pred} Centroid": normalize_vec(raw_centroid),
            "Your Input Parameters": normalize_vec(raw_user),
        }

        fig_radar = plot_radar_chart(
            categories=categories,
            cluster_means=radar_data,
            title=f"Multi-Feature Radar Comparison: User Input vs {climate_type}",
        )
        st.pyplot(fig_radar)
        plt.close(fig_radar)

    with tab3:
        st.subheader("Historical Climate Clusters Benchmark Directory")
        summary_rows = []
        for c_id, prof in profiles.items():
            c_entry = {
                "Cluster": int(c_id),
                "Regime Name": prof["label"],
                "Historical Share (%)": f"{prof['percentage']}%",
                "Mean Temp (°C)": prof["stats"]["Temperature"]["mean"],
                "Mean Humidity (%)": prof["stats"]["Humidity"]["mean"],
                "Mean Wind (km/h)": prof["stats"]["Wind_Speed"]["mean"],
                "Mean Pressure (hPa)": prof["stats"]["Pressure"]["mean"],
                "Mean Precip (mm)": prof["stats"]["Precipitation"]["mean"],
                "Hazard Level": prof["hazard_risk"],
            }
            summary_rows.append(c_entry)

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True)

    with tab4:
        st.subheader("Unsupervised Learning Architecture & Validation Metrics")
        st.markdown(
            f"""
            - **Champion Algorithm:** {metrics.get('champion_algorithm', 'K-Means')} (k={metrics.get('optimal_k', 5)})
            - **Silhouette Score:** `{metrics.get('kmeans_metrics', {}).get('silhouette_score', 'N/A')}` *(Measures cluster cohesion vs separation, range [-1, 1])*
            - **Calinski-Harabasz Index:** `{metrics.get('kmeans_metrics', {}).get('calinski_harabasz', 'N/A')}` *(Variance ratio criterion; higher indicates better partition)*
            - **Davies-Bouldin Index:** `{metrics.get('kmeans_metrics', {}).get('davies_bouldin', 'N/A')}` *(Similarity measure; lower indicates better clustering)*
            - **PCA Cumulative Explained Variance (2D):** `{metrics.get('pca_variance_ratio', {}).get('cumulative', 0.65):.1%}`
            - **Agglomerative Clustering Silhouette:** `{metrics.get('agglomerative_metrics', {}).get('silhouette_score', 'N/A')}`
            - **DBSCAN Discovered Clusters:** `{metrics.get('dbscan_metrics', {}).get('n_clusters_discovered', 'N/A')}` (Noise ratio: `{metrics.get('dbscan_metrics', {}).get('noise_ratio', 0.05):.1%}`)
            """
        )

        st.info(
            "**Why Feature Standardization is Mandatory:** "
            "Clustering algorithms (K-Means, Agglomerative, DBSCAN) rely on geometric distances (Euclidean metric). "
            "Atmospheric pressure (~1013 hPa) is orders of magnitude larger than precipitation (~5 mm) or wind speed (~15 km/h). "
            "Without `StandardScaler` (zero mean, unit variance), the pressure variable would dominate distance computations by over 99%, "
            "invalidating natural cluster boundaries."
        )


if __name__ == "__main__":
    main()
