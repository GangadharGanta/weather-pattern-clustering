# Weather Pattern Clustering for Localised Climate Insight Generation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://weather-pattern-clustering.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Unsupervised ML](https://img.shields.io/badge/Domain-Unsupervised%20Machine%20Learning-green.svg)](https://scikit-learn.org/)

An end-to-end, university-compliant Unsupervised Machine Learning repository developed for **INT396 (Unsupervised Machine Learning)**. The system clusters multi-variate historical weather telemetry into distinct synoptic climate regimes, evaluates cluster separation via rigorous mathematical validation metrics, flags localized atmospheric anomalies, and presents an interactive Streamlit dashboard.

---

## 📌 Project Metadata & Submission Links

| Metadata Field | Value |
| :--- | :--- |
| **Project Title** | Weather Pattern Clustering for Localised Climate Insight Generation |
| **Course Code / Subject** | INT396 (Unsupervised Machine Learning) |
| **Problem Statement ID** | 31 |
| **Student Name** | Ganta Gangadhar |
| **Student Registration Number** | 12523695 |
| **GitHub Repository** | [https://github.com/GangadharGanta/weather-pattern-clustering](https://github.com/GangadharGanta/weather-pattern-clustering) |
| **Live Deployed Web Application** | [https://weather-pattern-clustering.streamlit.app](https://weather-pattern-clustering.streamlit.app) |

---

## 📂 Project Directory Structure

```text
weather-pattern-clustering/
│
├── .gitignore                             # Ignores __pycache__, venv/, and data/raw/
├── requirements.txt                       # Pinned dependencies
├── PROJECT_LINKS.txt                      # Strict format university submission file
├── README.md                              # Technical guide & Viva Q&As
├── LICENSE                                # Open-source MIT license
│
├── notebooks/
│   └── weather_clustering_analysis.ipynb  # End-to-end academic EDA, PCA, & clustering notebook
│
├── data/
│   ├── raw/                               # Historical climate dataset
│   └── processed/                         # Scaled, cleaned, and normalized climate data
│
├── models/
│   ├── kmeans_champion.joblib             # Serialized champion clustering pipeline
│   ├── cluster_profiles.json              # Domain metadata and agricultural advisories
│   ├── training_metrics.json              # Quantitative benchmark scores (Inertia, Silhouette)
│   └── evaluation_plots/                  # High-resolution evaluation charts
│       ├── pca_clusters_2d.png            # 2D PCA latent cluster projections
│       ├── elbow_and_silhouette_tuning.png# Dual-axis WCSS elbow and silhouette curves
│       ├── radar_cluster_profiles.png     # Polar radar comparison across centroids
│       └── feature_distributions_by_cluster.png # Multi-panel feature boxplots
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data ingestion, synthetic generation, & cleaning
│   ├── feature_engineering.py             # StandardScaler, PCA, and cyclical transformers
│   ├── train.py                           # K-Means, Agglomerative, DBSCAN benchmarking & export
│   ├── evaluate.py                        # Cluster profiling and visual artifact generator
│   └── utils.py                           # Thermodynamics formulas & plotting helpers
│
├── app/
│   ├── app.py                             # Interactive Streamlit dashboard
│   └── assets/                            # Web application media
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                   # Automated Pytest suite (shapes, ranges, inference)
│
└── docs/
    ├── report_content.md                  # Comprehensive 10-section university project report draft
    ├── ppt_structure.md                   # Slide-by-slide presentation script (Links on 1 & 10)
    ├── PRESENTATION.pptx                  # Editable presentation slide deck
    └── PROJECT_REPORT.pdf                 # Formatted PDF project report
```

---

## 🔬 Mathematical Foundations & Methodology

### 1. Distance-Sensitive Feature Standardization
Unsupervised clustering algorithms (K-Means, DBSCAN, Agglomerative) compute proximity using Euclidean distance:
$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^D (x_{i,d} - x_{j,d})^2}$$
In raw meteorological telemetry, surface pressure ($\approx 1013\text{ hPa}$) is orders of magnitude larger than precipitation ($\approx 5\text{ mm}$) or wind speed ($\approx 15\text{ km/h}$). Without normalization, pressure would dominate the metric space by $>99\%$. `StandardScaler` standardizes each feature to zero mean ($\mu=0$) and unit variance ($\sigma=1$):
$$z = \frac{x - \mu}{\sigma}$$

### 2. Principal Component Analysis (PCA)
To project the 5-dimensional standardized space onto a 2D plane for visual inspection, we compute the eigenvectors of the empirical covariance matrix $\mathbf{\Sigma}$:
$$\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$$
The top two principal components capture over $65\%$ of the total atmospheric variance.

### 3. Hyperparameter Tuning & Validation Metrics
- **Elbow Method (WCSS / Inertia):** Measures within-cluster dispersion:
  $$\text{WCSS} = \sum_{k=0}^{K-1} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$
  Identifies the point of diminishing returns (elbow inflection at $k=5$).
- **Silhouette Coefficient ($s$):**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$
  Evaluates how cohesive an observation is within its cluster compared to neighboring clusters. Peak score ($0.4485$) is achieved at $k=5$.
- **Davies-Bouldin Index ($DB$):** Evaluates ratio of within-cluster spread to between-cluster separation. Minimized at $k=5$ ($0.8650$).

### 4. Continuous Anomaly Scoring
To flag localized extreme weather events or sensor drift, the system computes the standardized Euclidean distance from the observation $\mathbf{x}$ to its assigned centroid $\boldsymbol{\mu}_k$:
$$A(\mathbf{x}) = \|\text{Scaler}(\mathbf{x}) - \boldsymbol{\mu}_k\|_2$$
- $A(\mathbf{x}) < 2.0$: Typical meteorological pattern.
- $2.0 \le A(\mathbf{x}) < 3.5$: Moderate microclimatic deviation.
- $A(\mathbf{x}) \ge 3.5$: Extreme atmospheric anomaly.

---

## 🌤️ Discovered Climate Profiles & Domain Insights

| Cluster | Climate Profile Name | Synoptic Characteristics | Key Agronomic & Civil Advisory |
| :---: | :--- | :--- | :--- |
| **0** | **Hot & Humid (Tropical Monsoon)** | High Temp ($>30^\circ\text{C}$), High Humidity ($>80\%$), Low Pressure | Fungal disease alert for crops; heavy cooling & AC grid load. |
| **1** | **Cold & Dry (Winter Continental Fair)** | Low Temp ($<8^\circ\text{C}$), Low Humidity, High Pressure ($>1024\text{ hPa}$) | Ground frost warning for orchards; elevated domestic heating demand. |
| **2** | **Stormy & Unstable (Deep Depression Gale)** | Very Low Pressure ($<995\text{ hPa}$), Heavy Rain, High Winds ($>35\text{ km/h}$) | Flooding & lodging risk; suspend agricultural spraying; squall alert. |
| **3** | **Hot & Arid (Subtropical Heatwave)** | Extreme Temp ($>38^\circ\text{C}$), Low Humidity ($<25\%$), Desiccating Winds | Severe soil moisture depletion; acute irrigation demand; heatstroke alert. |
| **4** | **Pleasant & Temperate (Mild Spring/Autumn)** | Mild Temp ($20-24^\circ\text{C}$), Moderate Humidity ($50\%$), Stable Pressure | Optimal flowering and field conditions; minimal baseline HVAC load. |

---

## ⚡ Quickstart & Execution Guide

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/GangadharGanta/weather-pattern-clustering.git
cd weather-pattern-clustering

python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### 2. Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Training & Artifact Generation
```bash
# Ingest data, train K-Means/Agglomerative/DBSCAN, and export champion pipeline:
python src/train.py

# Generate evaluation charts (PCA 2D, Elbow, Radar, Boxplots):
python src/evaluate.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit Interactive Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501` to test interactive weather sliders and scenarios.

---

## 🎓 Viva Voce & Technical Defense Q&A

### Q1: Why is feature standardization (`StandardScaler`) mathematically mandatory for distance-based clustering?
**Answer:** Algorithms such as K-Means, Agglomerative Clustering, and DBSCAN compute proximity using the Euclidean $L_2$ norm:
$$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{d=1}^D (x_{id} - x_{jd})^2}$$
In meteorological datasets, atmospheric pressure is measured on an absolute scale of approximately $1013\text{ hPa}$, whereas precipitation is on a scale of $0 - 50\text{ mm}$. A small $1\%$ relative change in pressure produces a numerical difference of $10$, which squared is $100$. Meanwhile, a massive $100\%$ surge in rainfall from $2\text{ mm}$ to $4\text{ mm}$ produces a difference of $2$, which squared is only $4$. Without z-score standardization ($z = \frac{x - \mu}{\sigma}$), pressure would numerically dominate distance calculations by over $99\%$, rendering the model entirely blind to moisture, precipitation, and wind patterns.

---

### Q2: How does the Silhouette Coefficient differ from the Elbow Method (WCSS Inertia), and what do negative silhouette values indicate?
**Answer:** 
1. **WCSS (Inertia):** Calculates the total squared Euclidean distance between data points and their assigned cluster centroid:
   $$\text{WCSS} = \sum_{k=1}^K \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$
   WCSS monotonically decreases as $k$ increases, reaching $0$ when $k=N$. The Elbow Method seeks an empirical inflection point where adding another cluster yields diminishing returns.
2. **Silhouette Coefficient:** Evaluates both **cohesion** $a(i)$ (mean intra-cluster distance) and **separation** $b(i)$ (mean distance to the nearest neighboring cluster):
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
   Values range from $-1$ to $+1$. A positive score near $+1$ signifies compact, well-separated clusters. A score near $0$ indicates overlapping boundaries. A **negative silhouette score** ($s(i) < 0$) indicates that an observation is, on average, closer to points in a neighboring cluster than to points in its own assigned cluster, signifying misclassification or poor geometric cluster topology.

---

### Q3: Why is K-Means++ preferred over random centroid initialization, and what are its computational complexities?
**Answer:** Standard K-Means with random centroid selection is notoriously susceptible to poor local minima, which can produce suboptimal clusterings and slow convergence. K-Means++ resolves this by initializing the first centroid uniformly at random and subsequent centroids with probability proportional to the squared Euclidean distance from the nearest existing centroid:
$$P(\mathbf{x}) = \frac{D(\mathbf{x})^2}{\sum_{\mathbf{x}' \in \mathcal{X}} D(\mathbf{x}')^2}$$
This guarantees an expected clustering cost within $O(\log K)$ of the optimal clustering.
- **Training Time Complexity:** $O(I \cdot K \cdot N \cdot D)$, where $I$ is iterations ($\approx 10-30$), $K$ is number of clusters, $N$ is observations, and $D$ is feature dimension.
- **Inference Time Complexity:** $O(K \cdot D)$, requiring only $K$ distance calculations per new observation, allowing sub-millisecond real-time web deployment.

---

### Q4: When would DBSCAN be preferable to K-Means in meteorological clustering, and why did we choose K-Means as champion?
**Answer:** DBSCAN is superior when clusters exhibit arbitrary, non-spherical geometric shapes (e.g., elongated storm fronts) and when the primary objective is isolating anomalous noise points (extreme cyclonic squalls or faulty sensor readings) without forcing them into a cluster. 
However, for our localized climate operational dashboard, K-Means was selected as champion because:
1. Every incoming daily weather reading must be mapped to a known climate regime with structured agricultural and grid advisories (DBSCAN can reject points as noise with label `-1`).
2. K-Means maintains explicit centroids ($\boldsymbol{\mu}_k$), enabling continuous distance-based anomaly scoring ($A(\mathbf{x}) = \|\mathbf{x} - \boldsymbol{\mu}_k\|$).
3. K-Means supports instant $O(K \cdot D)$ inference on new streaming inputs without re-indexing all historical data.

---

### Q5: What is the Davies-Bouldin Index, and how does it validate your chosen $k=5$ topology?
**Answer:** The Davies-Bouldin (DB) Index is an internal clustering validation metric defined as:
$$DB = \frac{1}{K} \sum_{i=1}^K \max_{j \ne i} \left(\frac{R_i + R_j}{d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)}\right)$$
where $R_i$ represents the average distance of points in cluster $i$ to centroid $\boldsymbol{\mu}_i$ (cluster diameter/scatter), and $d(\boldsymbol{\mu}_i, \boldsymbol{\mu}_j)$ is the distance between centroids. Unlike WCSS, **lower values of Davies-Bouldin indicate better clustering**, signifying clusters that are internally compact and widely separated from each other. In our hyperparameter search, $k=5$ minimized the DB index to $0.8650$ (down from $1.341$ at $k=2$), corroborating the Silhouette score peak ($0.4485$) and validating the 5-cluster climate architecture.

---

## 📜 Academic Integrity & License
This project is submitted in partial fulfillment of the academic curriculum for INT396 at Lovely Professional University. Distributed under the MIT License.
