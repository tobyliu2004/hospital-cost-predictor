Hospital Cost Predictor
This project predicts hospital inpatient charges using structured medical and administrative data from the publicly available SPARCS dataset (New York State, 2011). It is designed as an industry-grade machine learning pipeline, incorporating modern practices in model training, explainability, fairness auditing, drift detection, and deployment. The project is fully reproducible and includes a deployable Streamlit application for live predictions and SHAP-based interpretation.

Table of Contents
Overview

Key Features

Data Source

Pipeline Architecture

Usage

1. Setup

2. Train the Model

3. Tune with Optuna

4. Launch the App

Streamlit App Features

Explainability with SHAP

Fairness and Bias Auditing

Drift Testing

Unsupervised Analysis

File Structure

Next Steps

Author

Overview
Predicting hospital charges is a critical task for improving transparency, fairness, and efficiency in healthcare systems. This project uses supervised learning to predict the logarithm of total inpatient charges and incorporates a full model development lifecycle:

Preprocessing and robust feature engineering

XGBoost model with Optuna hyperparameter tuning

Stratified regression splits for stable training

Explainability via SHAP

Fairness audits across race and gender

Drift testing for real-world simulation

Streamlit deployment for live predictions

Automated reporting with metrics and visualizations

Key Features
Model: XGBoost Regressor with log-transformed target

Deployment: Streamlit app with live predictions and SHAP explanations

Explainability: SHAP waterfall, summary, and bar plots

Audits: Fairness across demographic groups (groupwise MAE and residuals)

Robustness: Drift testing on simulated data shifts

Unsupervised Add-on: PCA, KMeans, Isolation Forest

Reporting: Markdown and PNG summaries auto-generated

Testing: Pytest coverage for pipeline execution

Data Source
The dataset comes from the SPARCS (Statewide Planning and Research Cooperative System) de-identified 2011 inpatient discharge data published by the New York State Department of Health.

Approx. 2.5 million rows (subsampled to 100,000 for development)

Includes demographic, diagnostic, procedural, admission/discharge, and financial fields

Pipeline Architecture
lua
Copy code
             +-----------------+
             | Raw CSV Data    |
             +--------+--------+
                      |
               [load_data()]
                      |
             +--------v--------+
             | clean_data()     | <-- Strip, coerce, log-transform
             +--------+--------+
                      |
               [preprocess()]
                      |
             +--------v--------+
             | Train/Test Split |
             +--------+--------+
                      |
               [train() + evaluate()]
                      |
             +--------v--------+
             |  XGBoost Model   |
             +--------+--------+
                      |
     +----------------v----------------+
     | SHAP | Fairness | Drift | Report |
     +----------------+----------------+
                      |
              [Streamlit App]
All components are organized across dedicated scripts and utility modules.

Usage
1. Setup
Install the required dependencies (preferably in a virtual environment):

bash
Copy code
pip install -r requirements.txt
Ensure the following directory structure is in place:

arduino
Copy code
project-root/
├── data/
│   └── hospital_data.csv
├── models/
├── reports/
├── images/
├── app.py
├── main.py
├── pipeline_trainer.py
├── config.py
├── scripts/
│   ├── tuning_utils.py
│   ├── explainability.py
│   ├── robustness_utils.py
│   ├── fairness_audit.py
│   ├── report_utils.py
│   ├── schema_utils.py
│   └── unsupervised_utils.py
└── tests/
    └── test_pipeline.py
2. Train the Model
You can run the full training pipeline using the following command:

bash
Copy code
python main.py
This will:

Load and clean the dataset

Preprocess features

Load best hyperparameters (from Optuna or a JSON file)

Train the XGBoost model

Evaluate model performance

Save the model to models/final_xgb_model.pkl

Generate SHAP values and save visualizations

Run fairness and drift audits

Output a Markdown report with key metrics

3. Tune with Optuna (Optional)
To search for the best XGBoost parameters:

python
Copy code
from scripts.tuning_utils import run_optuna
from config import DATA_PATH
import pandas as pd

from pipeline_trainer import BillingCostPredictor
model = BillingCostPredictor()
model.load_data(DATA_PATH)
model.clean_data()
model.preprocess()

best_params = run_optuna(model.X_train, model.y_train, n_trials=50)
4. Launch the App
To run the Streamlit app locally:

bash
Copy code
streamlit run app.py
The app allows you to:

Upload a CSV of patient records

View encoded features and predicted costs

Download predictions

Visualize feature importance using SHAP

Streamlit App Features
Live CSV upload and validation

Model introspection (expected feature list)

Interactive feature encoding preview

Live prediction with log-scale reversal to dollar amounts

SHAP waterfall explanation for first input row

CSV export of prediction results

Explainability with SHAP
SHAP (SHapley Additive exPlanations) is used to:

Compute per-feature contributions

Visualize feature importance across samples

Inspect individual predictions interactively (via waterfall plot)

Generated plots include:

shap_summary.png (summary bar chart)

shap_beeswarm.png (distribution of impact)

These are saved to the images/ directory and can be reused in reports or presentations.

Fairness and Bias Auditing
The model is evaluated for fairness using:

Groupwise MAE: Measures prediction error across sensitive groups (e.g., Race, Gender)

Residual Distributions: Checks if the model systematically over- or underpredicts for certain demographics

python
Copy code
from scripts.fairness_audit import compute_groupwise_mae, plot_groupwise_mae, residual_distribution_by_group

for col in ["Gender", "Race"]:
    print(f"Auditing {col}")
    mae_by_group = compute_groupwise_mae(df_test, y_true, y_pred, group_col=col)
    plot_groupwise_mae(mae_by_group, f"Groupwise MAE by {col}")
    residual_distribution_by_group(df_test, y_true, y_pred, group_col=col)
Drift Testing
To simulate real-world distributional shifts, this project includes a drift robustness module. For example, you can simulate an unseen insurance plan in the test set:

python
Copy code
from scripts.robustness_utils import simulate_drift, evaluate_on_drift

X_columns = model.X_test.columns
drift_values = ["CryptoCare PPO", "UnknownPlan", "OutOfNetwork"]

drift_results = []
for val in drift_values:
    df_drifted = simulate_drift(df_test, "Payment Typology 1", val)
    metrics = evaluate_on_drift(model.model, df_drifted, X_columns, y_true)
    drift_results.append(metrics)
Unsupervised Analysis
Using PCA and clustering, the project includes tools for visualizing hidden structure in the data and flagging outliers:

PCA: Reduce to 2D for visualization

KMeans: Cluster samples for cohort analysis

Isolation Forest: Flag anomalous samples that deviate from the norm

python
Copy code
from scripts.unsupervised_utils import run_unsupervised_audit, plot_pca_clusters, plot_pca_anomalies

audit_df, explained_var = run_unsupervised_audit(model.X_test)
plot_pca_clusters(audit_df)
plot_pca_anomalies(audit_df)
File Structure
graphql
Copy code
.
├── app.py                         # Streamlit frontend
├── main.py                        # Full model training + auditing pipeline
├── config.py                      # Centralized paths and column definitions
├── pipeline_trainer.py            # Class-based training pipeline
├── data.py                        # Data cleaning helpers
├── models/                        # Saved models and SHAP values
├── data/                          # Input CSV file (hospital_data.csv)
├── reports/                       # Auto-generated Markdown reports
├── images/                        # SHAP plots
├── scripts/
│   ├── explainability.py
│   ├── fairness_audit.py
│   ├── report_utils.py
│   ├── robustness_utils.py
│   ├── tuning_utils.py
│   ├── schema_utils.py
│   └── unsupervised_utils.py
├── tests/
│   └── test_pipeline.py           # Pytest-based validation
Next Steps
Add model versioning metadata via model cards

Package with CLI support (argparse) for full automation

Implement test coverage for Streamlit app and feature engineering

Connect to external databases or cloud storage for real-time prediction

Optionally deploy on Hugging Face Spaces, Streamlit Cloud, or AWS Lambda

Author
Toby Liu
Undergraduate Researcher, ML Engineer in training
LinkedIn | GitHub

This project was developed as a capstone for PSTAT 131 (Statistical Machine Learning) and further expanded as part of a professional-grade ML portfolio targeting industry roles.

For questions, suggestions, or collaborations, feel free to open an issue or reach out via GitHub or LinkedIn.

License: MIT (or specify your preferred license)

