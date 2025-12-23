# NYC Yellow Taxi Fare Predictor

## Project Overview
This project aims to predict the fare and duration of taxi trips in New York City to improve customer experience and transparency.By leveraging machine learning on historical trip data, this solution offers real-time estimates, allowing users to plan their trips and manage expenses effectively.

The project encompasses the full data science lifecycle: from exploratory data analysis and geospatial feature engineering to model optimization and deployment via a web dashboard.

## Authors
* **Louis Imbert** 
* **Njee Hettiarachchi** 
* **Marwan Hemani** 

---

## Repository Structure
The project is divided into three main notebooks available in the `main` branch:

* **`Data Visualization Final.ipynb`**: Contains the Exploratory Data Analysis (EDA).It covers univariate and bivariate analysis, handling of missing values, and outlier detection (e.g., negative fare amounts).
* **`Feature_Selection_Methods.ipynb`**: details the methods used to reduce dimensionality and select the most relevant variables using Filter methods like ANOVA F-test and Mutual Information.
* **`Model Training Final.ipynb`**: Includes the training pipeline, hyperparameter tuning (RandomSearchCV), and the final ensemble model creation using Tree-based algorithms.

---

## The Dataset
We utilized the **NYC Yellow Taxi Trip Data** from the NYC Taxi & Limousine Commission (TLC), specifically the Kaggle version for **January 2015**.
* **Volume:** 12.6+ million rows.
* **Key Features:** Pickup/Dropoff datetime, Passenger count, Trip distance, Longitude/Latitude, Fare amount.

---

## Methodology

### 1. Data Cleaning & Visualization
* **Outlier Removal:** Removed impossible values such as negative fares and unrealistic trip durations.
* **Imbalance Handling:** Analyzed payment types and passenger counts distributions.

### 2. Feature Engineering
To improve model accuracy, we generated new features:
* **Temporal Features:** Extracted `Day_of_Week` and `Moments_of_Day` to capture traffic trends.
* **Geospatial Clustering:** Used **KMeans clustering** (200 clusters) to group pickup/dropoff coordinates into "Neighborhood" areas. We utilized the **GeoPy API** with OpenStreetMap to label these centroids, avoiding the need for complex polygon mapping.

### 3. Feature Selection
We applied filter methods to rank feature importance:
* **ANOVA F-test:** For linear dependencies.
* **Mutual Information:** To capture non-linear dependencies.
* **Consensus:** Found `trip_distance`, `RateCodeID`, and spatial clusters to be the most critical.

### 4. Model Selection
Given the large dataset size, we focused on parallelizable Tree-based models:
* Random Forest Regressor
* XGBoost
* LightGBM
* CatBoost

---

## Results
After hyperparameter tuning using `RandomSearchCV`, we created a **Voting Classifier** ensemble combining our best performing models (Optimized LightGBM and CatBoost).

| Model | R² Score |
| :--- | :--- |
| Random Forest | 0.905 |
| XGBoost | 0.905 |
| **LightGBM** | **0.907** |
| **Voting Classifier (Final)** | **0.907** |


---

## Deployment (Dash App)
We developed a local dashboard using **Dash** to demonstrate the model's capabilities.
* **Features:**
    * User input for Departure/Arrival addresses (converted to coordinates via API).
    * Interactive Map displaying the route.
    * **Predictions:** Estimated Fare ($) and Trip Duration (min).

---

## 🛠️ Requirements
To run the notebooks, the following key libraries are required:
* Python 3.x
* Pandas / NumPy
* Scikit-learn
* XGBoost / LightGBM / CatBoost
* GeoPy
* Dash / Plotly (for the dashboard)
