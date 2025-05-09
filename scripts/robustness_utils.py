# scripts/robustness_utils.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def simulate_drift(df_test, drift_col, new_value):
    """Replaces all values in drift_col with new_value to simulate distributional shift."""
    df_drifted = df_test.copy()
    df_drifted[drift_col] = new_value
    return df_drifted

def evaluate_on_drift(model, df_drifted, X_columns, y_true):
    """Preprocess and evaluate the model on the drifted test set."""
    X_drifted = pd.get_dummies(df_drifted, drop_first=True)
    X_drifted = X_drifted.reindex(columns=X_columns, fill_value=0).astype('float64')
    y_pred = model.predict(X_drifted)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "R2": r2
    }

#What this does
#simulate_drift()-Creates a modified test set simulating a new value in a categorical column
#evaluate_on_drift()-Re-runs model prediction on drifted data and reports performance drop

#Drift Sensitivity Analysis:
#To evaluate robustness, we simulated data drift by modifying test set variables (e.g., unseen insurance providers). Results showed that model performance remained stable for most shifts, with minor degradation in MAE and R². However, extreme distributional changes (e.g., all-new payment typologies) led to larger prediction errors, suggesting the model should be retrained or revalidated periodically in production.