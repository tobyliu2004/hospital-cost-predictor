# main.py
#below to push new changes to github
#git add .
#git commit -m "Update README and project files"
#git push origin main
from pipeline_trainer import BillingCostPredictor
from scripts.tuning_utils import run_optuna #Dont need this because saved the run optuna to disk
from scripts.fairness_audit import compute_groupwise_mae, plot_groupwise_mae, residual_distribution_by_group
from scripts.robustness_utils import simulate_drift, evaluate_on_drift
from scripts.report_utils import (
    save_metrics_to_markdown,
    save_shap_summary_plot,
    save_shap_beeswarm_plot
)
from xgboost import XGBRegressor
import json
from pathlib import Path

from config import DATA_PATH, BEST_PARAMS_PATH, FINAL_MODEL_PATH, SHAP_VALUES_PATH, METRICS_MD_PATH, SHAP_SUMMARY_IMG, SHAP_BEESWARM_IMG
from config import MODELS_DIR, REPORTS_DIR, IMAGES_DIR
for path in [MODELS_DIR, REPORTS_DIR, IMAGES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

#import scripts_robustness_utils
#importlib.reload(scripts_robustness_utils)
#from scripts_robustness_utils import simulate_drift, evaluate_on_drift

#if you update a function call this to make it read it
import importlib
import pipeline_trainer
importlib.reload(pipeline_trainer)
from pipeline_trainer import BillingCostPredictor
model = BillingCostPredictor()
#or just restart the kernel by exiting and reentering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === STEP 1: Initialize Model ===
#WHEN FINALIZING MODEL, TRAIN ON FULl 250k DATASET WITH 50 trials
model = BillingCostPredictor()
df_full = pd.read_csv(DATA_PATH)
df_sample = df_full.sample(n=100000, random_state=42)
df_sample.to_csv("hunnidthou_df.csv", index=False)
model.load_data("/Users/tobyliu/PSTAT 131 Final Project/hunnidthou_df.csv")
model.clean_data()
model.preprocess()

# === STEP 2: Hyperparameter Optimization ===
#best_params = run_optuna(model.X_train, model.y_train, n_trials=30)
#SAVES best_params to disk
#with open("models/best_params.json", "w") as f:
#    json.dump(best_params, f)
#LOADS THE best_params from disk
with open(BEST_PARAMS_PATH) as f:
    best_params = json.load(f)
del model.model
model.model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)

print("✅ model type:", type(model.model))
import xgboost
print(xgboost.__version__)

# === STEP 3: Train + Evaluate ===
model.train()
results = model.evaluate()
model.save_model(FINAL_MODEL_PATH)

# === STEP 4: Explainability (SHAP) ===
from utils.explainability import run_shap, plot_shap_summary, plot_shap_beeswarm
explainer, shap_values = run_shap(model.model, model.X_train, model.X_test)

#saves explainer and shap_values to disk
#import joblib
# Save SHAP values (can be large, so use joblib)
#joblib.dump(shap_values, SHAP_VALUES_PATH)
#print("✅ SHAP values saved to models/shap_values.pkl")

#loads shap_values
import joblib
if SHAP_VALUES_PATH.exists():
    print("📦 Loading existing SHAP values...")
    shap_values = joblib.load(SHAP_VALUES_PATH)
else:
    print("🧠 Computing SHAP values from scratch...")
    explainer, shap_values = run_shap(model.model, model.X_train, model.X_test)
    joblib.dump(shap_values, SHAP_VALUES_PATH)
    print(f"✅ SHAP values saved to {SHAP_VALUES_PATH}")


plot_shap_summary(shap_values, model.X_test)
plot_shap_beeswarm(shap_values, model.X_test)

# === STEP 5: Fairness Audit ===
df_test = model.df.loc[model.X_test.index].copy()
y_true = model.y_test
y_pred = model.model.predict(model.X_test)

#THROW THIS IN THE JUPYTER NOTEBOOK
for col in ["Gender", "Race"]:
    print(f"🔍 Fairness Audit on {col}")
    mae_by_group = compute_groupwise_mae(df_test, y_true, y_pred, group_col=col)
    plot_groupwise_mae(mae_by_group, f"Groupwise MAE by {col}")
    residual_distribution_by_group(df_test, y_true, y_pred, group_col=col)

# === STEP 6: Drift Testing ===
X_columns = model.X_test.columns
drift_values = ["CryptoCare PPO", "UnknownPlan", "OutOfNetwork"]
drift_results = []

for val in drift_values:
    df_drifted = simulate_drift(df_test, "Payment Typology 1", val)
    metrics = evaluate_on_drift(model.model, df_drifted, X_columns, y_true)
    metrics["Drift Value"] = val
    drift_results.append(metrics)

pd.DataFrame(drift_results).set_index("Drift Value").plot(kind='bar', figsize=(8,6), title="Model Performance Under Simulated Drift")
plt.ylabel("Metric Score")
plt.tight_layout()
plt.show()

# === STEP 7: Auto-Reporting ===
save_metrics_to_markdown(results, output_path=METRICS_MD_PATH)
save_shap_summary_plot(shap_values, model.X_test, out_path=SHAP_SUMMARY_IMG)
save_shap_beeswarm_plot(shap_values, model.X_test, out_path=SHAP_BEESWARM_IMG)