# scripts/fairness_audit.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#compute_groupwise_mae-Are some groups getting worse predictions than others?
def compute_groupwise_mae(df, y_true, y_pred, group_col):
    df_temp = df.copy()
    df_temp['True'] = y_true
    df_temp['Pred'] = y_pred
    df_temp['Abs_Error'] = abs(df_temp['True'] - df_temp['Pred'])
    return df_temp.groupby(group_col)['Abs_Error'].mean().sort_values(ascending=False)

#plot_groupwise_mae-Visualize model accuracy inequality
def plot_groupwise_mae(mae_series, title):
    plt.figure(figsize=(10,6))
    sns.barplot(x=mae_series.values, y=mae_series.index, palette='mako')
    plt.title(title)
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Group")
    plt.tight_layout()
    plt.show()

#residual_distribution_by_group-See if model consistently over/underpredicts certain groups
def residual_distribution_by_group(df, y_true, y_pred, group_col):
    df_temp = df.copy()
    df_temp['Residual'] = y_true - y_pred
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df_temp, x=group_col, y='Residual')
    plt.axhline(0, linestyle='--', color='red')
    plt.title(f"Residual Distribution by {group_col}")
    plt.tight_layout()
    plt.show()
