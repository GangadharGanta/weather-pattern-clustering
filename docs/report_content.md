# Weather Pattern Clustering for Localised Climate Insight Generation
**Course:** INT396 (Unsupervised Machine Learning)  
**Problem Statement ID:** 31  
**Author:** Ganta Gangadhar (Registration Number: 12523695)  
**Project Links:**
- GitHub Repository: [https://github.com/GangadharGanta/weather-pattern-clustering](https://github.com/GangadharGanta/weather-pattern-clustering)
- Interactive Cloud Deployment: [https://weather-pattern-clustering.streamlit.app](https://weather-pattern-clustering.streamlit.app)

---

## Abstract
Local meteorological dynamics exhibit complex, non-linear dependencies governed by thermodynamics, solar insolation, and atmospheric circulation. Traditional weather forecasting relies heavily on numerical weather prediction (NWP) models that are computationally expensive and struggle with rapid, localized regime classification. This project develops an end-to-end unsupervised machine learning pipeline to segment continuous daily multi-variate weather observations into distinct, interpretable climate profiles. Utilizing features including Temperature, Humidity, Wind Speed, Atmospheric Pressure, and Precipitation, the system evaluates K-Means, Agglomerative Hierarchical Clustering, and Density-Based Spatial Clustering of Applications with Noise (DBSCAN). By leveraging StandardScaler and Principal Component Analysis (PCA) for latent manifold projection, hyperparameter optimization via the Elbow Method (WCSS) and Silhouette analysis identifies an optimal $k=5$ clustering topology. The resulting regimes—Hot & Humid, Cold & Dry, Stormy & Unstable, Hot & Arid, and Pleasant & Temperate—enable real-time anomaly detection, agricultural advisory synthesis, and energy grid load planning. The framework is deployed as an interactive Streamlit web dashboard.

---

## 1. Introduction & Background
Climate variability and localized weather patterns directly govern agricultural yields, renewable energy generation, water resource management, and civil infrastructure resilience. While synoptic-scale weather stations collect high-frequency time-series data, raw continuous readings often fail to provide rapid, high-level operational insights without contextual interpretation. 

Unsupervised machine learning provides a natural mathematical paradigm to discover latent structures within meteorological data without requiring costly ground-truth human annotations. By treating multi-dimensional atmospheric measurements as vectors in a metric space, clustering algorithms partition daily observations into distinct weather regimes. Each regime represents a recurring synoptic state (e.g., anticyclonic high-pressure fair weather versus cyclonic depression storms), translating numerical sensor telemetry into localized domain insights.

---

## 2. Problem Formulation & Objectives (Problem Statement ID: 31)
### 2.1 Problem Formulation
Given a continuous multi-dimensional time-series dataset of daily weather observations:
$$\mathcal{D} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}, \quad \mathbf{x}_i \in \mathbb{R}^D$$
where $D=5$ denotes the primary meteorological feature space:
$$\mathbf{x}_i = [\text{Temperature}_i, \text{Humidity}_i, \text{Wind Speed}_i, \text{Pressure}_i, \text{Precipitation}_i]^T$$
the objective is to learn an unsupervised partitioning function:
$$f: \mathbb{R}^D \to \{0, 1, \dots, K-1\}$$
such that within-cluster meteorological cohesion is maximized and inter-cluster separation is maximized, subject to distance metric constraints.

### 2.2 Core Objectives
1. **Automated Preprocessing & Imputation:** Build robust pipelines for chronological time-series imputation (forward-fill and backward-fill) and physical boundary validation.
2. **Standardization & Dimensionality Reduction:** Formulate a distance-preserving scaling pipeline with `StandardScaler` to prevent feature dominance (e.g., pressure in hPa versus precipitation in mm), followed by `PCA` for 2D latent visualization.
3. **Algorithm Exploration & Benchmarking:** Rigorously compare K-Means++, Ward's Agglomerative Hierarchical Clustering, and DBSCAN across internal clustering validity indices.
4. **Hyperparameter Optimization:** Quantitatively determine optimal cluster count $k$ using Inertia (WCSS Elbow curve) and the Silhouette Coefficient.
5. **Domain Profiling & Anomaly Formulation:** Generate interpretable climate profiles and formulate an Euclidean distance-to-centroid anomaly metric.
6. **Web Application Deployment:** Deliver an interactive, responsive Streamlit dashboard featuring live slider inputs, 2D PCA projections, and radar profiles.

---

## 3. Literature Survey & Related Work
Clustering techniques have been widely studied in meteorological and environmental engineering:
1. **Synoptic Typing via K-Means:** Philipp et al. (2010) demonstrated that K-Means clustering applied to atmospheric sea-level pressure fields effectively captures European circulation patterns with high computational efficiency.
2. **Hierarchical Approaches in Climatology:** Kalkstein et al. (1987) utilized average linkage and Ward's minimum variance method for automated weather type classification, noting strong spatial coherence but higher algorithmic complexity ($O(N^2)$).
3. **Density-Based Clustering for Extreme Weather:** Ester et al. (1996) and subsequent atmospheric studies highlighted DBSCAN's capability to isolate meteorological anomalies and irregularly shaped fronts as noise points without forcing them into spherical clusters.
4. **Principal Component Latent Embeddings:** Richman (1986) reviewed rotational and unrotated PCA in meteorology, proving that linear dimensionality reduction isolates dominant modes of variance while attenuating sensor noise.

---

## 4. Dataset Description & Exploratory Data Analysis (EDA)
### 4.1 Feature Space
The historical weather dataset spans 1,826 daily observations (5 complete annual cycles) encompassing the following core meteorological parameters:
- **Temperature ($^\circ\text{C}$):** Continuous daily mean ambient thermal reading.
- **Humidity ($\%$):** Relative atmospheric water vapor saturation.
- **Wind Speed ($\text{km/h}$):** Mean horizontal air velocity.
- **Atmospheric Pressure ($\text{hPa}$):** Barometric pressure at station surface level.
- **Precipitation ($\text{mm}$):** Cumulative 24-hour liquid precipitation.
- **Derived Features:** Dew Point ($^\circ\text{C}$) computed via the Magnus-Tetens relationship, and Apparent Temperature ($^\circ\text{C}$) reflecting thermal comfort.

### 4.2 Statistical Summary & Physical Correlations
- **Clausius-Clapeyron Thermodynamic Relationship:** Strong physical coupling is observed between temperature and moisture-carrying capacity. Elevated temperatures coincide with potential peaks in absolute humidity, whereas rapid barometric drops ($\le 1000\text{ hPa}$) correlate strongly with elevated precipitation and gusty wind conditions.
- **Seasonality:** Annual sinusoidal variation governs ambient temperature with maximums during summer months (June–August) and minimums in winter (December–February).

---

## 5. Methodology & Mathematical Framework

### 5.1 Distance-Sensitive Feature Standardization
Clustering algorithms depend fundamentally on geometric distances:
$$d(\mathbf{x}_i, \mathbf{x}_j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{d=1}^D (x_{i,d} - x_{j,d})^2}$$
Atmospheric pressure operates on the scale of $10^3\text{ hPa}$, whereas precipitation operates on the scale of $10^0 - 10^1\text{ mm}$. Without standardization, pressure differences would account for over $99\%$ of the computed Euclidean distance, rendering the clustering blind to humidity, wind, and rainfall.
Each feature is transformed via z-score standardization:
$$z_{i,d} = \frac{x_{i,d} - \mu_d}{\sigma_d}$$
where $\mu_d$ and $\sigma_d$ are the empirical mean and standard deviation of feature $d$.

### 5.2 Principal Component Analysis (PCA)
To project the standardized 5D data onto a 2D orthogonal subspace for visualization:
1. Compute the empirical covariance matrix:
   $$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{Z}^T \mathbf{Z}$$
2. Compute eigenvalues $\lambda_j$ and eigenvectors $\mathbf{v}_j$:
   $$\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$$
3. Form the projection matrix $\mathbf{W} = [\mathbf{v}_1, \mathbf{v}_2] \in \mathbb{R}^{5 \times 2}$. The resulting coordinates $\mathbf{Z}_{\text{pca}} = \mathbf{Z} \mathbf{W}$ capture approximately $65-70\%$ of total atmospheric variance across two principal axes.

### 5.3 Clustering Formulations
- **K-Means++:** Minimizes Within-Cluster Sum of Squares (Inertia):
  $$J = \sum_{k=0}^{K-1} \sum_{\mathbf{z}_i \in C_k} \|\mathbf{z}_i - \boldsymbol{\mu}_k\|^2$$
  Initialized via $D^2$ probability weighting to accelerate convergence and avoid degenerate local minima.
- **Agglomerative Hierarchical Clustering:** Employs Ward's linkage criterion minimizing variance increase at each agglomeration step:
  $$\Delta \text{ESS}_{A,B} = \frac{n_A n_B}{n_A + n_B} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$
- **DBSCAN:** Identifies density-reachable cores with parameters $\epsilon$ (neighborhood radius) and $\text{MinPts}$ (minimum density threshold), naturally classifying anomalous storms or sensor glitches as noise ($label = -1$).

---

## 6. Hyperparameter Optimization & Model Selection
Internal clustering validity was assessed across cluster counts $k \in [2, 8]$ using three metrics:

1. **Within-Cluster Sum of Squares (WCSS / Inertia):** Monotonically decreases as $k$ increases. The Elbow Method identifies the point of maximum diminishing returns (inflection point).
2. **Silhouette Coefficient ($s$):**
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s \in [-1, 1]$$
   where $a(i)$ is mean intra-cluster distance and $b(i)$ is mean nearest-cluster distance.
3. **Davies-Bouldin Index ($DB$):** Evaluates cluster similarity; lower values indicate tighter, more distinct clusters:
   $$DB = \frac{1}{K} \sum_{k=1}^K \max_{j \ne k} \left(\frac{R_k + R_j}{d(\boldsymbol{\mu}_k, \boldsymbol{\mu}_j)}\right)$$

### Benchmark Results Table
| Number of Clusters ($k$) | WCSS (Inertia) | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
| :---: | :---: | :---: | :---: | :---: |
| 2 | 5820.4 | 0.3340 | 1145.2 | 1.3410 |
| 3 | 4310.2 | 0.3780 | 1420.8 | 1.1205 |
| 4 | 3120.6 | 0.4190 | 1680.4 | 0.9840 |
| **5 (Optimal)** | **2240.1** | **0.4485** | **1985.6** | **0.8650** |
| 6 | 1980.5 | 0.4020 | 1810.2 | 0.9210 |
| 7 | 1790.3 | 0.3810 | 1715.4 | 0.9540 |

**Selection:** $k=5$ yields the highest Silhouette score ($0.4485$), optimal Davies-Bouldin minimization ($0.8650$), and clear physical meteorological separability.

---

## 7. Results & Meteorological Cluster Profiling

The five discovered clusters align with recognized synoptic weather regimes:

1. **Cluster 0 — Hot & Humid (Tropical / Sultry Monsoon):**
   - *Centroid:* Temp: $30.2^\circ\text{C}$, Humidity: $81.5\%$, Wind: $15.8\text{ km/h}$, Pressure: $1005.2\text{ hPa}$, Precip: $9.4\text{ mm}$.
   - *Domain Insights:* Elevated heat index; significant agricultural fungal disease vulnerability; elevated HVAC cooling loads.
2. **Cluster 1 — Cold & Dry (Winter Continental Fair):**
   - *Centroid:* Temp: $5.8^\circ\text{C}$, Humidity: $37.2\%$, Wind: $13.9\text{ km/h}$, Pressure: $1024.6\text{ hPa}$, Precip: $0.2\text{ mm}$.
   - *Domain Insights:* High atmospheric stability; clear skies; frost risk for sensitive horticultural crops; elevated domestic heating loads.
3. **Cluster 2 — Stormy & Unstable (Deep Depression / Gale):**
   - *Centroid:* Temp: $18.4^\circ\text{C}$, Humidity: $91.2\%$, Wind: $38.6\text{ km/h}$, Pressure: $993.4\text{ hPa}$, Precip: $34.8\text{ mm}$.
   - *Domain Insights:* Severe cyclonic activity; crop lodging and soil erosion risk; infrastructure alerts for wind and flood hazards.
4. **Cluster 3 — Hot & Arid (Subtropical Heatwave):**
   - *Centroid:* Temp: $39.5^\circ\text{C}$, Humidity: $23.1\%$, Wind: $21.4\text{ km/h}$, Pressure: $1010.8\text{ hPa}$, Precip: $0.0\text{ mm}$.
   - *Domain Insights:* High evapotranspiration rate; severe irrigation demand; heatstroke hazard; peak electrical grid stress.
5. **Cluster 4 — Pleasant & Temperate (Mild Spring / Autumn):**
   - *Centroid:* Temp: $21.8^\circ\text{C}$, Humidity: $51.4\%$, Wind: $11.8\text{ km/h}$, Pressure: $1015.4\text{ hPa}$, Precip: $0.4\text{ mm}$.
   - *Domain Insights:* Optimal outdoor thermal comfort; low baseline municipal energy demand; ideal agronomic conditions for flowering.

---

## 8. Real-Time Anomaly Scoring Formulation
In addition to categorical clustering, the system formulates a continuous localized anomaly metric:
$$A(\mathbf{x}) = \|\text{Scaler}(\mathbf{x}) - \boldsymbol{\mu}_{C(\mathbf{x})}^*\|_2$$
where $\boldsymbol{\mu}_{C(\mathbf{x})}^*$ is the scaled centroid of the assigned cluster $C(\mathbf{x})$.
- $A(\mathbf{x}) < 2.0$: Typical Observation (Statistically consistent with historical norms).
- $2.0 \le A(\mathbf{x}) < 3.5$: Moderate Deviation (Localized atmospheric instability or atypical microclimate).
- $A(\mathbf{x}) \ge 3.5$: Extreme Anomaly (Potential uncalibrated sensor, severe microburst, or unprecedented climate extreme).

---

## 9. Interactive Web Application Architecture (`app/app.py`)
The user interface is engineered in Streamlit, structured around three core operational components:
1. **Interactive Control Panel:** Dynamic sliders for real-time telemetry inputs (Temperature, Humidity, Wind Speed, Pressure, Precipitation) with one-click presets for rapid testing.
2. **Synchronous Inference Pipeline:** Loads serialized artifacts (`models/kmeans_champion.joblib`) to execute standardization, cluster assignment, PCA projection, and anomaly scoring in $<15\text{ ms}$.
3. **Visual Analytics Suite:**
   - Real-time 2D PCA projection highlighting the active input with a distinct golden star against the historical cluster manifold.
   - Normalized multi-feature radar chart comparing the user's input directly against the cluster centroid.
   - Interactive data tables detailing historical cluster metrics and meteorological advisories.

---

## 10. Conclusion, Limitations & Future Scope
### 10.1 Conclusion
This project successfully demonstrates that unsupervised clustering (K-Means++ with $k=5$, paired with `StandardScaler` and `PCA`) partitions complex multi-variate weather data into distinct, physically meaningful climate regimes with a Silhouette score of $0.4485$. The system delivers automated, interpretable climate insights for agriculture, renewable energy, and emergency response.

### 10.2 Limitations
- **Spatial Stationarity:** Current clustering operates on single-station localized time-series; spatial gradients between adjacent geographic stations are not modeled.
- **Linear Manifold Assumption:** PCA captures linear variance; highly non-linear atmospheric manifold structures may require Kernel PCA or t-SNE / UMAP for non-linear embedding.

### 10.3 Future Scope
- **Spatio-Temporal Integration:** Incorporate Graph Convolutional Networks (GCNs) or Recurrent Autoencoders (LSTM/GRU Autoencoders) to cluster multi-station temporal sequences.
- **Automated Weather Alerts:** Integrate SMS/Webhook dispatching when anomaly scores exceed critical thresholds ($A(\mathbf{x}) > 3.5$).
