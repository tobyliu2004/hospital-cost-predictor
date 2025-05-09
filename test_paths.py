# test_paths.py
from config import *

def check_paths():
    print("🔍 Verifying config paths:")
    for name, path in {
        "DATA_PATH": DATA_PATH,
        "MODEL_PATH": FINAL_MODEL_PATH,
        "BEST_PARAMS_PATH": BEST_PARAMS_PATH,
        "SHAP_VALUES_PATH": SHAP_VALUES_PATH,
        "METRICS_MD_PATH": METRICS_MD_PATH,
        "SHAP_SUMMARY_IMG": SHAP_SUMMARY_IMG,
        "SHAP_BEESWARM_IMG": SHAP_BEESWARM_IMG
    }.items():
        exists = path.exists()
        print(f"- {name}: {'✅ Exists' if exists else '⚠️ Missing'} → {path}")

if __name__ == "__main__":
    check_paths()
