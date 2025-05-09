# pipeline/trainer.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
import joblib
from config import FINAL_MODEL_PATH, REQUIRED_COLUMNS
from utils.schema import validate_input_schema

class BillingCostPredictor:
    def __init__(self, model=None, model_name="XGBoost", test_size=0.2, random_state=42):
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = model if model else XGBRegressor(random_state=42, n_jobs=-1)
        self.fitted = False

    def load_data(self, path):
        self.df = pd.read_csv(path)
        print(f"✅ Data loaded: {self.df.shape}")
        return self.df

    def clean_data(self):
        # 🚨 Absolute guarantee this is a DataFrame
        assert isinstance(self.df, pd.DataFrame), "self.df is not a DataFrame"
        df = self.df.copy(deep=True)
        df = df.drop(columns=['Total Costs'], errors='ignore')
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
        print("✅ Passed: df is a DataFrame")

        # Print column names to confirm load worked
        print("🔎 First 5 column names:", df.columns.tolist()[:5])
        print("🔎 df shape:", df.shape)

        # === Clean string columns early to avoid weird behaviors
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()

        # === Force known numeric columns to float before any math
        numeric_cols = [
            'Length of Stay',
            'APR Risk of Mortality',
            'APR Severity of Illness Code',
            'Total Costs',
            'Total Charges',
            'Birth Weight',
            'Age Group Ordinal'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # === Create log-transformed target (if present and numeric) ===
        # From here: proceed with log transform and feature engineering...
        if 'Total Charges' in df.columns:
            df['Log Total Charges'] = np.log1p(df['Total Charges'])

        # === Validate types before proceeding
        for col in ['APR Severity of Illness Code', 'Length of Stay', 'APR Risk of Mortality']:
            dtype = df[col].dtype
            print(f"🔍 {col} dtype: {dtype}")
            assert dtype in ['float64', 'int64'], f"{col} is not numeric"

        # If you make it this far: safe to continue
        print("✅ All numeric types validated")

        self.df = df
        print(f"✅ Data cleaned and assigned: {df.shape}")
        return df

    def preprocess(self, drop_columns=['Total Charges', 'Log Total Charges', 'Total Costs']):
        X = self.df.drop(columns=drop_columns, errors = 'ignore')
        y = self.df['Log Total Charges']

        # Encode
        X = pd.get_dummies(X, drop_first=True)

        # Clean column names to make them XGBoost-safe
        X.columns = X.columns.astype(str).str.replace(r"[^a-zA-Z0-9_]", "_", regex=True)

        # Bin target into quantiles for stratified regression
        y_binned = pd.qcut(y, q=10, duplicates='drop')

        # Stratified Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y_binned)

        # Align
        self.X_train, self.X_test = self.X_train.align(self.X_test, join='left', axis=1, fill_value=0)

        # Ensure float
        self.X_train = self.X_train.astype('float64')
        self.X_test = self.X_test.astype('float64')

        print(f"✅ Preprocessing done. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self):
        print("🧪 Training with", type(self.model))
        self.model.fit(self.X_train, self.y_train, verbose=True)
        self.fitted = True
        print("✅ Model trained.")


    def evaluate(self, cv=5):
        if not self.fitted:
            raise ValueError("Model not trained yet.")

        preds = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, preds)
        rmse = np.sqrt(mean_squared_error(self.y_test, preds))
        r2 = r2_score(self.y_test, preds)

        cv_mae = -np.mean(cross_val_score(
            self.model, self.X_train, self.y_train,
            scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1
        ))

        print(f"📊 {self.model_name} Performance:")
        print(f"  Test MAE:  {mae:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  Test R²:   {r2:.4f}")
        print(f"  CV MAE:    {cv_mae:.4f}")

        return {
            'Test MAE': mae,
            'Test RMSE': rmse,
            'Test R²': r2,
            'CV MAE': cv_mae
        }

    def save_model(self, path=FINAL_MODEL_PATH):
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")