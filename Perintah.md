Create a complete, end-to-end industrial Machine Learning Jupyter Notebook named `mining_predictive_maintenance.ipynb` using Python. The entire notebook—including all markdown documentation cells, technical explanations, and code comments—must be written in English.

Execute and implement the following machine learning pipeline steps:

### 1. Environment Setup & Data Ingestion
- Import core dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost`.
- Load the dataset directly from the local workspace file: `predictive_maintenance.csv`.
- Inspect schema, data types, missing values, and print dataset dimensions (`shape`).

### 2. Data Cleaning & Feature Engineering
- Drop non-predictive ID columns: `UDI` and `Product ID`.
- Drop specific failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) to prevent target label leakage into `Machine failure`.
- Encode the equipment quality variant column `Type` ordinally: `{'L': 0, 'M': 1, 'H': 2}`.
- Construct 3 mechanical domain-specific features:
  1. Mechanical Power (W): `Torque [Nm] * Rotational speed [rpm] * (2 * pi / 60)`
  2. Delta Temperature (K): `Process temperature [K] - Air temperature [K]`
  3. Wear Strain Index: `Torque [Nm] * Tool wear [min]`

### 3. Stratified Partitioning & Leakage-Free Scaling
- Separate feature matrix `X` and target vector `y` (`Machine failure`).
- Perform an 80:20 Stratified Train-Test Split with `stratify=y` and `random_state=42` to maintain the ~3.4% failure class distribution.
- Apply `StandardScaler`: strictly call `.fit_transform()` on `X_train` and `.transform()` on `X_test` to prevent data leakage.

### 4. Ensemble Modeling with Class Imbalance Mitigation
- Train Model 1: Random Forest Classifier with `class_weight='balanced'`, `n_estimators=150`, `max_depth=12`, `random_state=42`, `n_jobs=-1`.
- Train Model 2: XGBoost Classifier with `scale_pos_weight` set to `count(negatives) / count(positives)`, `n_estimators=150`, `learning_rate=0.05`, `max_depth=5`, `eval_metric='logloss'`, `random_state=42`, `n_jobs=-1`.

### 5. Quantitative Evaluation & Visual Diagnostics
- Evaluate both models on the test set using Confusion Matrix, Recall, Precision, F1-Score, and ROC-AUC.
- Generate and display visual plots:
  - Side-by-side Confusion Matrix heatmaps using `seaborn`.
  - Comparative ROC Curves annotated with AUC scores.
  - Feature Importance bar charts highlighting primary failure drivers (`Torque`, `Rotational speed`, and engineered interaction features).
- Save generated evaluation charts as high-resolution PNG images (`model_evaluation_metrics.png`, `feature_importance_analysis.png`).

### 6. Business Impact & Risk Predictions Export
- Include an operational markdown analysis explaining how early detection of torque and rotational velocity anomalies prevents unscheduled downtime and saves heavy mining maintenance costs.
- Compute continuous failure probabilities and map them into operational risk categories:
  - Low Risk (< 0.25)
  - Moderate Risk (0.25 to 0.65)
  - High Risk (>= 0.65)
- Export the test set predictions (including actual status, predicted status, failure probability, and risk tier) into `predictive_maintenance_risk_predictions.csv`.

Verify and ensure the generated `mining_predictive_maintenance.ipynb` file executes without error and is saved directly in the project workspace.