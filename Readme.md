# House Price Prediction

This project predicts California house prices using machine learning. It utilizes the **California Housing dataset** from Scikit-learn, performs **data preprocessing, exploratory data analysis (EDA), model training, and evaluation**, and saves the trained model for deployment.

---

## 📂 Project Overview
- **Goal:** Predict the median house value in California districts (in $100,000s) based on demographic and geographic data.
- **Dataset:** [California Housing Dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) (from Scikit-learn)
- **Algorithm Used:** `GradientBoostingRegressor`
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `pickle`

---

## 📊 Steps in the Project
1. **Data Loading**
   - Loaded the California housing dataset using `fetch_california_housing()`.
   - Created a DataFrame for analysis and added the target column `Prize` (median house value).

2. **Exploratory Data Analysis (EDA)**
   - Summary statistics using `df.describe()`.
   - Checked for null values.
   - Analyzed correlations between features (e.g., `AveRooms` and `AveBedrms` were highly correlated).
   - Visualized data using **correlation heatmaps, boxplots, and KDE plots**.

3. **Data Preprocessing**
   - Split into **features (X)** and **target (y)**.
   - Performed **train-test split** (50% test size).
   - Standardized features using `StandardScaler`.

4. **Model Training**
   - Trained a `GradientBoostingRegressor`.
   - Evaluated using:
     - **Mean Squared Error (MSE):** `0.284`
     - **Mean Absolute Error (MAE):** `0.368`
     - **R² Score:** `0.785`

5. **Model Saving**
   - Saved the trained model as `model.pkl` using `pickle`.
   - Model can be loaded for predictions without retraining.
