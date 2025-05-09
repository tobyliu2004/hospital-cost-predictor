# scripts_unsupervised_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# === 1. PCA Dimensionality Reduction ===
def run_pca(X, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X)
    df_pca = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_components)], index=X.index)
    explained_var = pca.explained_variance_ratio_
    return df_pca, explained_var

# === 2. KMeans Clustering ===
def run_kmeans(X, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X)
    return pd.Series(cluster_labels, index=X.index, name="Cluster")

# === 3. Isolation Forest for Anomaly Detection ===
def run_anomaly_detection(X, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)
    is_outlier = iso.predict(X) == -1
    df_out = pd.DataFrame({
        "Anomaly_Score": scores,
        "Is_Anomaly": is_outlier.astype(int)
    }, index=X.index)
    return df_out

# === 4. Combined Audit Utility ===
def run_unsupervised_audit(X, n_clusters=5, contamination=0.01):
    df_pca, explained_var = run_pca(X, n_components=2)
    df_clusters = run_kmeans(X, n_clusters=n_clusters)
    df_anomalies = run_anomaly_detection(X, contamination=contamination)

    audit_df = pd.concat([X.copy(), df_pca, df_clusters, df_anomalies], axis=1)
    return audit_df, explained_var

# === 5. Visualization Utilities ===
def plot_pca_clusters(audit_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=audit_df,
        x="PC1", y="PC2",
        hue="Cluster",
        palette="Set2",
        s=60,
        edgecolor="black"
    )
    plt.title("PCA Projection with KMeans Clusters")
    plt.tight_layout()
    plt.show()

def plot_pca_anomalies(audit_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=audit_df,
        x="PC1", y="PC2",
        hue="Is_Anomaly",
        palette={0: 'lightgray', 1: 'red'},
        s=60,
        edgecolor="black"
    )
    plt.title("PCA Projection with Anomalies (Isolation Forest)")
    plt.tight_layout()
    plt.show()

def find_optimal_k(X, max_k=10):
    from sklearn.metrics import silhouette_score
    scores = []
    for k in range(2, max_k+1):
        labels = KMeans(n_clusters=k, random_state=42, n_init='auto').fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append((k, score))
    return scores
