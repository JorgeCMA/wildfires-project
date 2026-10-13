Of course. Based on the provided notebooks, here is a detailed plan for an unsupervised learning phase, designed to uncover hidden structures and patterns within the wildfire dataset.

***

# **Unsupervised Learning Plan for Wildfire Data**

This document outlines a plan to apply unsupervised machine learning techniques to the cleaned and enriched wildfire dataset (`data/processed/expanded/firms_spain_final_lags_index.csv`). The goal is to move beyond pre-defined categories and discover the natural groupings, anomalies, and underlying structures within the data. This exploratory phase can provide valuable insights for feature engineering, model selection, and overall understanding of wildfire dynamics.

## **1. Data Source and Preparation**

**Input File:** `data/processed/expanded/firms_spain_final_lags_index.csv`
This file contains the fully prepared dataset, including:
- Spatial, temporal, and sensor data (FRP, brightness, confidence).
- Detailed meteorological variables from Open-Meteo (temperature, wind, humidity, VPD, soil moisture, etc.).
- Terrain data from CLC (land cover classes, neighbor classes).
- Engineered features: temporal cycles, meteorological indices, and human activity proxies.

**Key Considerations:**
- **Feature Scaling:** Most unsupervised algorithms are distance-based. All numerical features must be scaled (e.g., using `StandardScaler` or `MinMaxScaler`). `StandardScaler` is preferred as it is less sensitive to outliers.
- **Dimensionality Reduction:** The dataset has high dimensionality (57 columns). PCA, as shown in the notebooks, is a good start. For more complex, non-linear relationships, UMAP (Uniform Manifold Approximation and Projection) should be strongly considered as a pre-processing step or for visualization.

## **2. Phase 1: Clustering Wildfire Events**

The primary objective is to identify distinct "types" or "regimes" of fires based on their characteristics. We are not just clustering by location, but by the *nature* of the event itself.

### **2.1. Feature Selection for Clustering**

For clustering, it's best to start with a focused set of features that describe the fire's behavior and its environmental context.

- **Fire Intensity & Behavior:**
    - `frp` (Fire Radiative Power): The key measure of virulence.
    - `brightness`: Thermal intensity at the time of detection.
- **Meteorological Drivers ("Fire Weather"):**
    - `temperature_2m`: Heat.
    - `relative_humidity_2m`: Dryness.
    - `wind_speed_10m`: Wind-driven propagation.
    - `precipitation`: Immediate fuel wetness.
    - `vapour_pressure_deficit` (VPD): Atmospheric dryness.
    - `soil_moisture_0_to_7cm`: Fuel moisture content.
- **Topography & Fuel:**
    - `clc_class`: The dominant land cover type.
- **Temporal Context:**
    - `month_sin` / `month_cos`: Seasonality.
    - `franja_horaria`: Time of day (categorical, would need encoding).
- **Human Factors:**
    - `riesgo_antropico_alto`: Proxy for human presence.

### **2.2. Proposed Algorithms**

1.  **K-Means Clustering:**
    - **Approach:** A robust and fast baseline. We will run it on the scaled features.
    - **Methodology:** Use the **Elbow Method** and **Silhouette Analysis** to determine the optimal number of clusters (k). We can start by testing k in the range of 3-10.
    - **Expected Outcome:** "Fire regimes" such as "High-Wind, Low-Humidity Fires," "Summer Human-Caused Fires," "Low-Intensity Spring Fires," etc. This will help us understand which combinations of environmental conditions are most common.

2.  **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
    - **Approach:** DBSCAN does not require specifying the number of clusters and can identify outliers (noise) that do not belong to any cluster.
    - **Methodology:** This is an excellent approach for this dataset because not all fires fit neatly into a regime. Some are extreme, rare events. We will need to tune `eps` (epsilon) and `min_samples` parameters.
    - **Expected Outcome:** Identification of "anomalous" or "extreme" fire events that are structurally different from the main clusters.

3.  **Gaussian Mixture Models (GMM):**
    - **Approach:** A probabilistic model that assumes all data points are generated from a mixture of a finite number of Gaussian distributions. Unlike K-Means, it provides the probability of a point belonging to a cluster.
    - **Methodology:** Fit GMMs with different numbers of components and compare them using the Bayesian Information Criterion (BIC).
    - **Expected Outcome:** A soft-clustering of fires, where we can identify mixed-nature fires and have a measure of confidence in the cluster assignment.

### **2.3. Visualization and Interpretation**

- **Dimensionality Reduction for Visualization:** UMAP or t-SNE will be used to project the high-dimensional clustered data into 2D or 3D space. This allows us to visualize the clusters and see how they relate to each other.
- **Visualizing the Clusters:** The identified clusters will be plotted on the 2D projection, colored by their cluster label.
- **Profile Analysis:** For each cluster, we will compute the mean (or median) of key features (e.g., a bar chart for the mean `frp`, `temperature`, `wind_speed`, `precipitation`, and the distribution of `clc_class`) to build a "profile" that characterizes each fire regime.

## **3. Phase 2: Anomaly and Outlier Detection**

This phase focuses exclusively on finding rare and potentially catastrophic events.

### **3.1. Proposed Algorithms**

1.  **Isolation Forest:**
    - **Approach:** This algorithm works by creating random decision trees and "isolating" observations. Anomalies are points that are isolated in fewer splits, making them "easy to isolate."
    - **Methodology:** Fit an Isolation Forest on a broad set of scaled features (similar to those used for clustering). The `contamination` parameter will be tuned to a small value (e.g., 0.01 to 0.05) to define the expected proportion of outliers.
    - **Expected Outcome:** A flag (1 or -1) for each wildfire event, identifying it as an anomaly or not.

2.  **Local Outlier Factor (LOF):**
    - **Approach:** LOF measures the local density deviation of a given data point with respect to its neighbors. A lower density than its neighbors indicates an outlier.
    - **Methodology:** This is useful for finding outliers that might not be globally extreme, but are unusual in their specific local context (e.g., a moderately intense fire in a region where all fires are typically small).
    - **Expected Outcome:** A score for each point, where a significantly higher score indicates an anomaly.

### **3.2. Analysis of Anomalies**

- **Characterization:** We will characterize these anomalous events by analyzing their statistical difference from the rest of the dataset.
- **Spatiotemporal Analysis:** We will investigate *where* and *when* these anomalies occur. Are they clustered in certain CCAA? Do they happen at specific times of the year? Do they correlate with certain land cover types?
- **Link to FRP:** The most intense fires (highest FRP) are expected to be flagged as anomalies. We will verify this and use it to understand what combination of features leads to high virulence.

## **4. Phase 3: Advanced Pattern Discovery (Optional)**

### **4.1. Self-Organizing Maps (SOM)**

- **Approach:** SOM is a type of neural network used for dimensionality reduction and clustering. It maps high-dimensional data onto a low-dimensional (typically 2D) grid while preserving the topological relationships between data points. It is a powerful visualization and clustering tool.
- **Methodology:** Train a SOM on the same scaled features used for clustering. The result is a grid where similar fire events are mapped to nearby neurons.
- **Expected Outcome:** A two-dimensional "map" of the fire types. We can then cluster the neurons themselves and analyze the feature profiles of different areas of the map to discover a more nuanced set of fire regimes.

## **5. Expected Deliverables and Actionable Insights**

- **New Feature `fire_regime`:** The cluster labels from the best-performing model (e.g., K-Means or GMM) can be added to the dataset as a new categorical feature. This new feature can then be used in downstream *supervised* learning models to predict fire severity or behavior.
- **New Feature `is_anomaly`:** A boolean flag from the anomaly detection models, marking each event as an outlier or not.
- **Actionable Insights:** The project team will gain a deeper understanding of:
    - The distinct types of wildfires that occur in Spain.
    - The most dangerous combinations of meteorological and topographical conditions.
    - A list of the most extreme, atypical fire events for targeted case studies.
    - Whether the existing data captures the full spectrum of fire behavior or if we are blind to certain types of fires (e.g., smoldering peat fires versus fast-moving crown fires).
- **Visualizations:** A comprehensive report with visualizations (UMAP plots, cluster profiles, anomaly maps) to communicate the findings to the project stakeholders.