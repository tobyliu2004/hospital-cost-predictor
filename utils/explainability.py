import shap
import matplotlib.pyplot as plt

#USED IN main.py

#initializes s SHAP explainer using trained XGBoost model
#calculates SHAP values for the test set(X_test)
#returns explainer and SHAP values
#SHAP values tell you how much each feature contributed to each prediction
def run_shap(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    return explainer, shap_values

#plots a bar chart showing the top contributing features
#based on mean absolute SHAP values
#helps see which features are globally most important in your model
def plot_shap_summary(shap_values, X, top_n=20):
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=top_n)

#creates besswarm plot showing how each feature affects predictions accross samples
#color = feature value, position = SHAP value
def plot_shap_beeswarm(shap_values, X):
    shap.summary_plot(shap_values, X)
