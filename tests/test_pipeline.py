import pytest
from pipeline_trainer import BillingCostPredictor
from config import DATA_PATH

def test_pipeline_runs_without_error():
    model = BillingCostPredictor()
    model.load_data(DATA_PATH)
    model.clean_data()
    X_train, X_test, y_train, y_test = model.preprocess()
    
    assert X_train.shape[0] > 0
    assert list(X_train.columns) == list(X_test.columns)

def test_model_training_and_eval():
    model = BillingCostPredictor()
    model.load_data(DATA_PATH)
    model.clean_data()
    model.preprocess()
    model.train()
    metrics = model.evaluate()
    
    assert metrics["Test MAE"] > 0
    assert metrics["Test R²"] <= 1