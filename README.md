# 🏥 Hospital Cost Predictor

This is a professional-grade machine learning project built to predict hospital inpatient charges using SPARCS 2011 discharge data from New York State.

It includes:
- ✅ XGBoost regression pipeline with Optuna tuning
- 📊 SHAP explainability and feature importance visualizations
- ⚖️ Fairness audit across race/gender
- 🌪️ Drift testing with simulated insurance shifts
- 🔍 Unsupervised anomaly detection and PCA clustering
- 📦 Streamlit app for public use and CSV upload

---

## 🔧 Features

| Module              | Description |
|---------------------|-------------|
| `pipeline_trainer.py` | Full class-based training pipeline |
| `main.py`            | End-to-end training + evaluation + explainability |
| `app.py`             | Streamlit frontend for public predictions |
| `utils/`             | Modular tools: SHAP, fairness, drift, tuning, preprocessing |

---

## 🚀 Try the App (hosted on Streamlit)

➡️ [https://yourusername.streamlit.app](https://yourusername.streamlit.app) *(placeholder link)*

---

## 🧪 Local Setup

```bash
git clone https://github.com/yourusername/hospital-cost-predictor.git
cd hospital-cost-predictor
pip install -r requirements.txt
streamlit run app.py