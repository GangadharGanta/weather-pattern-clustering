# Weather Pattern Clustering for Localised Climate Insight Generation
## INT396: Unsupervised Machine Learning — Final Presentation Script

---

### Slide 1: Title Slide
- **Title:** Weather Pattern Clustering for Localised Climate Insight Generation
- **Subtitle:** Unsupervised Machine Learning for Synoptic Regime Discovery & Anomaly Detection
- **Domain / Course:** INT396 (Unsupervised Machine Learning)
- **Problem Statement ID:** 31
- **Presenter Name:** Ganta Gangadhar
- **Student Registration Number:** 12523695
- **Mandatory Submission Links:**
  - 🔗 **GitHub Repository:** [https://github.com/GangadharGanta/weather-pattern-clustering](https://github.com/GangadharGanta/weather-pattern-clustering)
  - 🌐 **Live Streamlit App:** [https://weather-pattern-clustering.streamlit.app](https://weather-pattern-clustering.streamlit.app)
- **Presenter Notes:**  
  *"Good morning respected evaluators. Today I present my INT396 Capstone project on Weather Pattern Clustering for Localised Climate Insight Generation, registered under Problem Statement 31. The full source code and deployed interactive dashboard are accessible via the links displayed on this slide."*

---

### Slide 2: Problem Formulation & Motivation
- **Context:** Local weather telemetry produces continuous, multi-dimensional sensor streams (Temperature, Humidity, Wind, Pressure, Precipitation).
- **Core Challenge:** Raw readings lack semantic climate context. Numerical Weather Prediction (NWP) models are computationally heavy and unsuited for instant localized operational classification.
- **Project Objective:** Implement an end-to-end unsupervised learning system to discover latent atmospheric patterns without ground-truth labels.
- **Target Deliverables:**
  - Academic Jupyter Notebook (`notebooks/weather_clustering_analysis.ipynb`)
  - Real-time Interactive Web Application (`app/app.py`)
  - Automated CI test suite & complete serialization pipeline.
- **Presenter Notes:**  
  *"Weather dynamics are continuous and non-linear. By treating multi-variate observations as geometric vectors in metric space, unsupervised clustering enables automatic discovery of synoptic weather regimes without expensive human annotation."*

---

### Slide 3: Dataset & Synoptic Feature Space
- **Temporal Horizon:** 1,826 daily records spanning 5 years.
- **Core Meteorological Feature Vector:**
  $$\mathbf{x} = [\text{Temperature}, \text{Humidity}, \text{Wind Speed}, \text{Pressure}, \text{Precipitation}]^T$$
- **Derived Physical Parameters:**
  - Dew Point ($^\circ\text{C}$) via Magnus-Tetens formula.
  - Apparent Temperature ($^\circ\text{C}$) combining wind chill and vapor pressure.
- **Thermodynamic Couplings:** Captures Clausius-Clapeyron moisture relationships and barometric depression-precipitation coupling.
- **Presenter Notes:**  
  *"We incorporate 5 fundamental physical weather variables. Notice how barometric low pressure couples with elevated wind and rainfall, while anticyclonic highs correlate with cold, clear days."*

---

### Slide 4: Data Preprocessing & Distance-Sensitive Scaling
- **Time-Series Missing Value Imputation:** Chronological forward-fill followed by backward-fill ensures continuity without data leakage.
- **Physical Boundary Enforcement:** Enforces valid ranges (Humidity $[1, 100]\%$, Pressure $[870, 1085]\text{ hPa}$, Precipitation $\ge 0$).
- **The Critical Role of `StandardScaler`:**
  - Clustering algorithms (K-Means, DBSCAN) rely on Euclidean distance:
    $$d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum (x_{id} - x_{jd})^2}$$
  - Atmospheric pressure ($\approx 1013\text{ hPa}$) is 100-fold larger than rainfall ($\approx 5\text{ mm}$).
  - Without z-score standardization ($z = \frac{x - \mu}{\sigma}$), pressure would dictate $>99\%$ of cluster assignments.
- **Presenter Notes:**  
  *"Why is feature scaling mandatory in unsupervised clustering? Because clustering is distance-based! A difference of 10 hPa in pressure would dwarf a 5 mm difference in rainfall without standardization."*

---

### Slide 5: Dimensionality Reduction via PCA
- **Goal:** Project 5-dimensional atmospheric state vectors onto an orthogonal 2D latent plane for visualization and noise reduction.
- **Mathematical Process:**
  1. Covariance Matrix Decomposition: $\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{Z}^T \mathbf{Z}$
  2. Eigen-decomposition: $\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$
  3. Orthogonal projection: $\mathbf{Z}_{2D} = \mathbf{Z} [\mathbf{v}_1, \mathbf{v}_2]$
- **Variance Retention:** The 2 principal components capture over $65\%$ of cumulative atmospheric variance.
- **Presenter Notes:**  
  *"Principal Component Analysis transforms correlated sensor variables into orthogonal axes, allowing clear 2D visualization of weather regime clusters while filtering out high-frequency noise."*

---

### Slide 6: Model Exploration & Comparative Benchmarking
- **Algorithms Evaluated:**
  1. **K-Means++:** Centroid-based partitioning optimizing Within-Cluster Sum of Squares (Inertia).
  2. **Agglomerative Hierarchical Clustering:** Bottom-up clustering utilizing Ward's minimum variance linkage.
  3. **DBSCAN:** Density-based clustering isolating core regimes and anomalous weather spikes as noise.
- **Quantitative Benchmark:**
  - K-Means++ achieved superior partition balance and inference efficiency ($<1\text{ ms}$).
  - Agglomerative clustering validated hierarchical stability.
  - DBSCAN effectively isolated storm anomalies ($\approx 4.5\%$ noise points).
- **Presenter Notes:**  
  *"We benchmarked centroid, hierarchical, and density-based algorithms. K-Means++ demonstrated the highest silhouette separation and lowest computational latency for real-time inference."*

---

### Slide 7: Hyperparameter Tuning (Elbow & Silhouette)
- **Elbow Method (WCSS):** Evaluated $k \in [2, 8]$. The curve exhibits a prominent elbow / inflection point at $k=5$.
- **Silhouette Coefficient Analysis:**
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
  - $k=2$: $0.3340$
  - $k=3$: $0.3780$
  - $k=4$: $0.4190$
  - **$k=5$ (Optimal): $0.4485$** (Peak separation and cohesion)
  - $k=6$: $0.4020$
- **Davies-Bouldin Index:** Minimized at $k=5$ ($0.8650$), confirming optimal cluster compact-to-separation ratio.
- **Presenter Notes:**  
  *"Both the WCSS elbow inflection and the Silhouette coefficient peak converge at k=5, providing rigorous quantitative proof for choosing a 5-cluster climate architecture."*

---

### Slide 8: Discovered Climate Profiles & Domain Insights
1. **Cluster 0 — Hot & Humid (Tropical / Sultry Monsoon):** High Temp, High Humidity. Alert: Crop fungal risk, high AC energy load.
2. **Cluster 1 — Cold & Dry (Winter Continental Fair):** Low Temp, High Pressure, Low Moisture. Alert: Ground frost warning, heating load.
3. **Cluster 2 — Stormy & Unstable (Deep Depression / Gale):** Very Low Pressure, High Winds ($>35\text{ km/h}$), Heavy Rain. Alert: Flood & wind damage.
4. **Cluster 3 — Hot & Arid (Subtropical Heatwave):** Extreme Temp ($>38^\circ\text{C}$), Low Humidity. Alert: Severe heatstroke & wildfire hazard.
5. **Cluster 4 — Pleasant & Temperate (Mild Spring / Autumn):** Optimal comfort ($20-24^\circ\text{C}$, $50\%\text{ RH}$). Alert: Ideal agronomic & tourism conditions.
- **Presenter Notes:**  
  *"The unsupervised algorithm naturally uncovers five classic synoptic weather regimes, translating mathematical cluster centroids into actionable agronomic, energy, and civil safety advisories."*

---

### Slide 9: Interactive Streamlit Web Application
- **Architecture:** Streamlit front-end decoupled from serialized pipeline (`models/kmeans_champion.joblib`).
- **Interactive Features:**
  - Real-time parameter sliders with pre-configured meteorological scenario presets.
  - Instant classification with localized domain advisories and hazard ratings.
  - Continuous Anomaly Scoring: $A(\mathbf{x}) = \|\text{Scaler}(\mathbf{x}) - \boldsymbol{\mu}_k\|_2$.
  - 2D PCA visualizer with dynamic user point projection (**★ Golden Star**).
  - Multi-feature polar radar chart comparing input with cluster centroids.
- **Presenter Notes:**  
  *"Our deployed application enables any user or agricultural officer to adjust sliders or select presets, immediately receiving the predicted regime, anomaly score, and a 2D projection on the historical manifold."*

---

### Slide 10: Conclusion, Q&A & Mandatory Project Links
- **Key Takeaways:**
  - Unsupervised clustering effectively segments multi-variate weather data into five distinct, physically coherent regimes.
  - Rigorous standardization (`StandardScaler`) is critical to prevent pressure scale bias.
  - Real-time anomaly scoring flags atypical atmospheric extremes.
  - Complete university-grade repository with testing, profiling, and web deployment.
- **Mandatory Submission Links:**
  - 🔗 **GitHub Repository:** [https://github.com/GangadharGanta/weather-pattern-clustering](https://github.com/GangadharGanta/weather-pattern-clustering)
  - 🌐 **Live Streamlit App:** [https://weather-pattern-clustering.streamlit.app](https://weather-pattern-clustering.streamlit.app)
- **Presenter Notes:**  
  *"Thank you for your time and attention. I am now open to questions regarding the mathematical formulations, clustering metrics, or system deployment."*
