import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def optuna_objective(trial, X, y, cv=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10)
    }

    model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    score = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    return -np.mean(score)

def run_optuna(X, y, n_trials=100, cv=5):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: optuna_objective(trial, X, y, cv), n_trials=n_trials)

    print("✅ Best Trial:")
    print("  MAE:", study.best_value)
    print("  Params:", study.best_params)
    return study.best_params

#What This Code Does: optuna_objective(...)-Tells Optuna how to train XGBoost and evaluate MAE on each trial
#run_optuna(...)-Launches the optimization search (default: 30 trials)
#It’s modular and reusable — works for any dataset.

