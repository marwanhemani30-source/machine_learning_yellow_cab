# NYC Yellow Taxi Fare Predictor

## Project Overview
[cite_start]This project aims to predict the fare and duration of taxi trips in New York City to improve customer experience and transparency[cite: 9, 11]. [cite_start]By leveraging machine learning on historical trip data, this solution offers real-time estimates, allowing users to plan their trips and manage expenses effectively[cite: 10].

The project encompasses the full data science lifecycle: from exploratory data analysis and geospatial feature engineering to model optimization and deployment via a web dashboard.

## Authors
* [cite_start]**Louis Imbert** [cite: 2]
* [cite_start]**Njee Hettiarachchi** [cite: 2]
* [cite_start]**Marwan Hemani** [cite: 2]

---

## Repository Structure
The project is divided into three main notebooks available in the `main` branch:

* **`Data Visualization Final.ipynb`**: Contains the Exploratory Data Analysis (EDA). [cite_start]It covers univariate and bivariate analysis, handling of missing values, and outlier detection (e.g., negative fare amounts)[cite: 64, 107].
* [cite_start]**`Feature_Selection_Methods.ipynb`**: details the methods used to reduce dimensionality and select the most relevant variables using Filter methods like ANOVA F-test and Mutual Information[cite: 260, 261].
* [cite_start]**`Model Training.ipynb`**: Includes the training pipeline, hyperparameter tuning (RandomSearchCV), and the final ensemble model creation using Tree-based algorithms[cite: 271, 282].

---

## The Dataset
[cite_start]We utilized the **NYC Yellow Taxi Trip Data** from the NYC Taxi & Limousine Commission (TLC), specifically the Kaggle version for **January 2015**[cite: 13, 16].
* [cite_start]**Volume:** 12.6+ million rows[cite: 55].
* [cite_start]**Key Features:** Pickup/Dropoff datetime, Passenger count, Trip distance, Longitude/Latitude, Fare amount[cite: 19, 29, 31, 42].

---

## Methodology

### 1. Data Cleaning & Visualization
* [cite_start]**Outlier Removal:** Removed impossible values such as negative fares and unrealistic trip durations[cite: 107, 250].
* [cite_start]**Imbalance Handling:** Analyzed payment types and passenger counts distributions[cite: 76].

### 2. Feature Engineering
To improve model accuracy, we generated new features:
* [cite_start]**Temporal Features:** Extracted `Day_of_Week` and `Moments_of_Day` to capture traffic trends[cite: 191].
* **Geospatial Clustering:** Used **KMeans clustering** (200 clusters) to group pickup/dropoff coordinates into "Neighborhood" areas. [cite_start]We utilized the **GeoPy API** with OpenStreetMap to label these centroids, avoiding the need for complex polygon mapping[cite: 209, 243].

### 3. Feature Selection
We applied filter methods to rank feature importance:
* [cite_start]**ANOVA F-test:** For linear dependencies[cite: 261].
* [cite_start]**Mutual Information:** To capture non-linear dependencies[cite: 264].
* [cite_start]**Consensus:** Found `trip_distance`, `RateCodeID`, and spatial clusters to be the most critical[cite: 269].

### 4. Model Selection
[cite_start]Given the large dataset size, we focused on parallelizable Tree-based models[cite: 282]:
* Random Forest Regressor
* XGBoost
* LightGBM
* CatBoost

---

## Results
[cite_start]After hyperparameter tuning using `RandomSearchCV` [cite: 312][cite_start], we created a **Voting Classifier** ensemble combining our best performing models (Optimized LightGBM and CatBoost)[cite: 320].

| Model | R² Score |
| :--- | :--- |
| Random Forest | 0.905 |
| XGBoost | 0.905 |
| **LightGBM** | **0.907** |
| **Voting Classifier (Final)** | **0.907** |

[cite_start][cite: 324, 332]

---

## Deployment (Dash App)
[cite_start]We developed a local dashboard using **Dash** to demonstrate the model's capabilities[cite: 366].
* **Features:**
    * User input for Departure/Arrival addresses (converted to coordinates via API).
    * [cite_start]Interactive Map displaying the route[cite: 375].
    * [cite_start]**Predictions:** Estimated Fare ($) and Trip Duration (min)[cite: 388, 389].

---

## 🛠️ Requirements
To run the notebooks, the following key libraries are required:
* Python 3.x
* Pandas / NumPy
* Scikit-learn
* XGBoost / LightGBM / CatBoost
* GeoPy
* Dash / Plotly (for the dashboard)
