# scripts/report_utils.py

import matplotlib.pyplot as plt
import shap
import os
import pandas as pd
from datetime import datetime

def save_metrics_to_markdown(metrics: dict, output_path="reports/metrics_report.md"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Model Evaluation Report\n\n")
        f.write(f"🕒 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in metrics.items():
            f.write(f"- **{key}**: {value:.4f}\n")

def save_shap_summary_plot(shap_values, X, out_path="images/shap_summary.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()

def save_shap_beeswarm_plot(shap_values, X, out_path="images/shap_beeswarm.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.clf()

#save_metrics_to_markdown()-Auto-writes a clean .md summary file
#save_shap_summary_plot()-Saves standard SHAP summary plot
#save_shap_beeswarm_plot()-Saves bar-style SHAP feature importance chart

#Automated Reporting & Reproducibility:
#All model metrics and plots are automatically saved during training to Markdown and PNG files. This enables reproducibility and consistent experiment tracking. Reports include MAE, RMSE, and R², as well as SHAP plots showing global feature importance. These logs can be version-controlled and shared with stakeholders or included in model validation pipelines.

