# Hospital Cost Predictor

This project predicts **hospital inpatient charges** using structured medical and administrative data from the publicly available **SPARCS dataset** (New York State, 2011). It is designed as an **industry-grade machine learning pipeline**, incorporating modern practices in model training, explainability, fairness auditing, drift detection, and deployment.

The project is fully reproducible and includes a deployable **Streamlit application** for live predictions and **SHAP-based interpretation**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Data Source](#data-source)
- [Pipeline Architecture](#pipeline-architecture)
- [Usage](#usage)
  - [1. Setup](#1-setup)
  - [2. Train the Model](#2-train-the-model)
  - [3. Tune with Optuna](#3-tune-with-optuna)
  - [4. Launch the App](#4-launch-the-app)
- [Streamlit App Features](#streamlit-app-features)
- [Explainability with SHAP](#explainability-with-shap)
- [Fairness and Bias Auditing](#fairness-and-bias-auditing)
- [Drift Testing](#drift-testing)
- [Unsupervised Analysis](#unsupervised-analysis)
- [File Structure](#file-structure)
- [Next Steps](#next-steps)
- [Author](#author)
- [License](#license)

---

## Overview

Predicting hospital charges is a critical task for improving transparency, fairness, and efficiency in healthcare systems. This project uses supervised learning to predict the **logarithm of total inpatient charges** and incorporates a full model development lifecycle:

- Preprocessing and robust feature engineering  
- XGBoost model with Optuna hyperparameter tuning  
- Stratified regression splits for stable training  
- Explainability via SHAP  
- Fairness audits across race and gender  
- Drift testing for real-world simulation  
- Streamlit deployment for live predictions  
- Automated reporting with metrics and visualizations  

---

## Key Features

- **Model**: XGBoost Regressor with log-transformed target  
- **Deployment**: Streamlit app with live predictions and SHAP explanations  
- **Explainability**: SHAP waterfall, summary, and bar plots  
- **Audits**: Fairness across demographic groups (groupwise MAE and residuals)  
- **Robustness**: Drift testing on simulated data shifts  
- **Unsupervised Add-on**: PCA, KMeans, Isolation Forest  
- **Reporting**: Markdown and PNG summaries auto-generated  
- **Testing**: Pytest coverage for pipeline execution  

---

## Data Source

- **Dataset**: SPARCS (Statewide Planning and Research Cooperative System), NY State Department of Health  
- **Size**: ~2.5 million rows (subsampled to 100,000 for development)  
- **Contents**: Demographics, diagnoses, procedures, admissions/discharges, and financial fields  

---

## Pipeline Architecture

```
Raw CSV Data
      |
  load_data()
      |
  clean_data()  <-- Strip, coerce, log-transform
      |
  preprocess()
      |
Train/Test Split
      |
train() + evaluate()
      |
XGBoost Model
      |
+---------------------------+
| SHAP | Fairness | Drift  |
+---------------------------+
         |
     Streamlit App
```

---

## Usage

### 1. Setup

```bash
pip install -r requirements.txt
```

Ensure the following directory structure:

```
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
```

### 2. Train the Model

```bash
python main.py
```

This will:
- Load and clean the dataset
- Preprocess features
- Load best hyperparameters
- Train the XGBoost model
- Save model and SHAP values
- Run audits and generate reports

### 3. Tune with Optuna

```python
from scripts.tuning_utils import run_optuna
from pipeline_trainer import BillingCostPredictor
from config import DATA_PATH

model = BillingCostPredictor()
model.load_data(DATA_PATH)
model.clean_data()
model.preprocess()

best_params = run_optuna(model.X_train, model.y_train, n_trials=50)
```

### 4. Launch the App

```bash
streamlit run app.py
```

---

## Streamlit App Features

- CSV upload for live prediction  
- Model feature introspection  
- Encoded feature previews  
- Prediction with log-reversal  
- SHAP waterfall and summary plots  
- CSV export of results  

---

## Explainability with SHAP

SHAP is used to:
- Interpret feature contributions  
- Visualize top drivers (bar, beeswarm, waterfall)  
- Improve model transparency  

Generated plots (in `/images/`):
- `shap_summary.png`
- `shap_beeswarm.png`

---

## Fairness and Bias Auditing

```python
from scripts.fairness_audit import (
    compute_groupwise_mae,
    plot_groupwise_mae,
    residual_distribution_by_group
)

for col in ["Gender", "Race"]:
    mae_by_group = compute_groupwise_mae(df_test, y_true, y_pred, group_col=col)
    plot_groupwise_mae(mae_by_group, f"Groupwise MAE by {col}")
    residual_distribution_by_group(df_test, y_true, y_pred, group_col=col)
```

---

## Drift Testing

```python
from scripts.robustness_utils import simulate_drift, evaluate_on_drift

X_columns = model.X_test.columns
for val in ["CryptoCare PPO", "UnknownPlan", "OutOfNetwork"]:
    df_drifted = simulate_drift(df_test, "Payment Typology 1", val)
    metrics = evaluate_on_drift(model.model, df_drifted, X_columns, y_true)
```

---

## Unsupervised Analysis

```python
from scripts.unsupervised_utils import (
    run_unsupervised_audit,
    plot_pca_clusters,
    plot_pca_anomalies
)

audit_df, _ = run_unsupervised_audit(model.X_test)
plot_pca_clusters(audit_df)
plot_pca_anomalies(audit_df)
```

---

## File Structure

```
.
├── app.py
├── main.py
├── config.py
├── pipeline_trainer.py
├── data.py
├── models/
├── data/
├── reports/
├── images/
├── scripts/
├── tests/
```

---

## Next Steps

- Add model card metadata  
- CLI argument support via `argparse`  
- Expand test coverage (Streamlit + preprocessing)  
- Optional cloud deployment (e.g., Render, Hugging Face Spaces)  
- Integration with MLflow or DVC  

---

## Author

**Toby Liu**  
Undergraduate Researcher, ML Engineer in Training  
[LinkedIn](https://www.linkedin.com/in/toby-liu-b45090257) | [GitHub](https://github.com/tobyliu2004)

This project was developed as a capstone for **PSTAT 131** and extended into a professional ML portfolio piece targeting industry roles.

---

## License

This project is licensed under the **MIT License**.
