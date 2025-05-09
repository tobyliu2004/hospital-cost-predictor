from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import joblib

#quantitative summary of how well your model performs on both train and test sets
#used in main.py
def evaluate_model(name, model, X_train, y_train, X_test, y_test, cv=5):
    """
    Trains and evaluates a model with optional cross-validation.
    Returns: metrics dict
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Core test set metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Cross-validated MAE on train
    cv_scores = cross_val_score(model, X_train, y_train, 
                                 scoring='neg_mean_absolute_error', 
                                 cv=cv, n_jobs=-1)
    
    print(f"\n🔍 Model: {name}")
    print(f" Test MAE:  {mae:.4f}")
    print(f" Test RMSE: {rmse:.4f}")
    print(f" Test R²:   {r2:.4f}")
    print(f" CV MAE (avg): {-np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    return {
        "Model": name,
        "Test MAE": mae,
        "Test RMSE": rmse,
        "Test R²": r2,
        "CV MAE": -np.mean(cv_scores),
        "CV MAE Std": np.std(cv_scores)
    }

#runs GridSearchCV over XGBoost hyperparameters
#uses neg MAE as the scoring metric
#returns best fit XGBRegressor based on cross validation
#used before optuna was made, still useful to keep
def tune_xgboost(X_train, y_train, param_grid=None, cv=5):
    from xgboost import XGBRegressor

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1],
        }

    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(xgb, param_grid, scoring='neg_mean_absolute_error', cv=cv, verbose=1, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("✅ Best Params:", grid.best_params_)
    print("✅ Best CV MAE:", -grid.best_score_)

    return grid.best_estimator_

#Saves trained model to disk using joblib
#creates directory models/ if doesnt exist, saves model and is loaded into streamlit app
def save_model(model, path="models/final_xgb_model.pkl"):
    import joblib
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")