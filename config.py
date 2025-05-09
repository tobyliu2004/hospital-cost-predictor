# config.py

from pathlib import Path

# === Root Paths ===
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
IMAGES_DIR = ROOT_DIR / "images"
REQUIRED_COLUMNS = [
    'Length of Stay',
    'Birth Weight',
    'Total Costs',
    'Total Charges',
    'APR Severity of Illness Code',
    'APR Risk of Mortality',
    'Age Group',
    'Age Group Ordinal',
    'APR Medical Surgical Description',
    'Type of Admission',
    'Emergency Department Indicator'
]


# === File Paths ===
DATA_PATH = DATA_DIR / "hospital_data.csv"  # or hunnidthou_df.csv
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"
FINAL_MODEL_PATH = MODELS_DIR / "final_xgb_model.pkl"
SHAP_VALUES_PATH = MODELS_DIR / "shap_values.pkl"
METRICS_MD_PATH = REPORTS_DIR / "final_metrics.md"
SHAP_SUMMARY_IMG = IMAGES_DIR / "shap_summary.png"
SHAP_BEESWARM_IMG = IMAGES_DIR / "shap_beeswarm.png"
