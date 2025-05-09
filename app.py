import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
from streamlit_shap import st_shap
from config import FINAL_MODEL_PATH, REQUIRED_COLUMNS
from utils.schema import validate_input_schema


# === Preprocessing Function ===
def preprocess_uploaded_data(df_raw, model):
    validate_input_schema(df_raw, REQUIRED_COLUMNS)
    drop_columns = ['Total Charges', 'Log Total Charges', 'Total Costs']
    expected_cols = model.get_booster().feature_names

    df = df_raw.copy()
    df = df.drop(columns=drop_columns, errors='ignore')
    df.columns = df.columns.str.strip()

    # === Safe type conversion ===
    df = df.convert_dtypes()
    df['Length of Stay'] = pd.to_numeric(df.get('Length of Stay'), errors='coerce')
    df['Birth Weight'] = pd.to_numeric(df.get('Birth Weight'), errors='coerce')
    df['Total Costs'] = pd.to_numeric(df.get('Total Costs'), errors='coerce')
    df['APR Severity of Illness Code'] = pd.to_numeric(df.get('APR Severity of Illness Code'), errors='coerce')
    df['APR Risk of Mortality'] = pd.to_numeric(df.get('APR Risk of Mortality'), errors='coerce')
    df['Age Group Ordinal'] = pd.to_numeric(df.get('Age Group Ordinal'), errors='coerce')

    # === FEATURE ENGINEERING (must match training) ===
    df['Log_Length_of_Stay'] = np.log1p(df['Length of Stay'])
    df['Log_Birth_Weight'] = np.log1p(df['Birth Weight'])

    df['Is_Extended_Stay'] = (df['Length of Stay'] > 7).astype(int)
    df['Is_Newborn'] = (df['Age Group'] == '0 to 17').astype(int)
    df['Is_Surgical'] = (df['APR Medical Surgical Description'] == 'Surgical').astype(int)
    df['Was_Emergency'] = (df['Type of Admission'] == 'Emergency').astype(int)
    df['Used_ER'] = (df['Emergency Department Indicator'] == 'Y').astype(int)

    df['Severity_LOS'] = df['APR Severity of Illness Code'] * df['Length of Stay']
    df['Risk_Cost'] = df['APR Risk of Mortality'] * df['Total Costs']
    df['Age_LOS'] = df['Age Group Ordinal'] * df['Length of Stay']
    df['Cost_per_Day'] = df['Total Costs'] / df['Length of Stay'].clip(lower=1)

    if 'Facility ID' in df.columns:
        freq_map = df['Facility ID'].value_counts(normalize=True).to_dict()
        df['Facility_Frequency'] = df['Facility ID'].map(freq_map)

    # === One-hot encoding + column cleanup ===
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded.columns = df_encoded.columns.astype(str).str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # === Align with model features ===
    df_aligned = df_encoded.reindex(columns=expected_cols, fill_value=0)
    return df_aligned.astype('float64')

def preprocess_like_training(df, drop_columns, expected_cols):
    # Drop target and unused columns
    df = df.drop(columns=drop_columns, errors='ignore')

    # One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Clean column names
    df_encoded.columns = df_encoded.columns.astype(str).str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

    # Align with model features
    df_encoded = df_encoded.reindex(columns=expected_cols, fill_value=0)
    return df_encoded.astype('float64')

# === Streamlit Setup ===
st.set_page_config(page_title="Hospital Charge Predictor", layout="centered")

st.title("\U0001F3E5 Hospital Cost Predictor")
st.write("Upload one or more rows of patient data as a CSV.")

# === Load model ===
model = joblib.load(FINAL_MODEL_PATH)
expected_cols = model.get_booster().feature_names
st.subheader("🧠 Model Feature Columns (Expected)")
st.write(expected_cols)

# === Upload CSV ===
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)
    st.subheader("\U0001F4CB Uploaded Data")
    st.dataframe(input_df)

    # === Preprocess input ===
    X_input = preprocess_uploaded_data(input_df, model)
    nonzero_cols = X_input.loc[:, (X_input != 0).any(axis=0)].columns.tolist()
    st.write("🧠 Non-zero features in this row:", nonzero_cols)

    st.write("🔍 Preview encoded input:")
    st.dataframe(X_input.head())

    # === Check input validity ===
    if (X_input != 0).sum().sum() == 0:
        st.error("❌ All input features are zero after encoding. Please ensure your CSV contains all necessary categories.")
        st.stop()

    st.write("\u2705 Input shape matches expected:", X_input.shape == (len(input_df), len(model.get_booster().feature_names)))
    st.write("\U0001F4DD Non-zero features per row:")
    st.dataframe((X_input != 0).sum(axis=1).rename("Non-zero count"))

    # === Predict ===
    pred_log = model.predict(X_input)
    pred_dollars = np.expm1(pred_log).round(2)

    results_df = input_df.copy()
    results_df["Predicted Charges ($)"] = pred_dollars

    st.subheader("\U0001F52E Predicted Charges")
    st.dataframe(results_df)

    # === SHAP explanation (only for first row) ===
    with st.expander("🔍 See SHAP explanation for first row"):
        explainer = shap.Explainer(model, X_input)
        shap_values = explainer(X_input)
        st_shap(shap.plots.waterfall(shap_values[0], max_display=15), height=500)

import io
csv_buffer = io.StringIO()
results_df.to_csv(csv_buffer, index=False)
csv_data = csv_buffer.getvalue()

st.download_button(
    label="📁 Download Predictions as CSV",
    data=csv_data,
    file_name="predicted_charges.csv",
    mime="text/csv"
)



# dummy change
