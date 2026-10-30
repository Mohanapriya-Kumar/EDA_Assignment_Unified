"""
Fall Semester 2026-2027
Digital Assignment 
Name: Mohanapriya Kumar
Reg. No: 23BDS0108
Course Code: BCSE331L
Course Name: Exploratory Data Analysis 
Course Type: Th
Class Number: \tVL2026270102447
Slot: A2
"""

# ASSESSMENT 8: Clustering Analysis

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, spearmanr, pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42
HAPPINESS_URL = "https://raw.githubusercontent.com/salemprakash/EDA/main/Data/happiness.csv"

def section(title):
    """Print a clear heading before each experiment."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def load_happiness_data():
    """Load happiness.csv and normalize blank cells to missing values."""
    data = pd.read_csv(HAPPINESS_URL)
    return data.replace(r"^\s*$", np.nan, regex=True)

def mode_value(series):
    """Return the most frequent non-missing value (the mode)."""
    modes = series.dropna().mode()
    return modes.iloc[0] if not modes.empty else np.nan

def first_numeric_column(frame, preferred=None):
    """Select a numeric column, preferring the requested name if available."""
    if preferred and preferred in frame.columns and pd.api.types.is_numeric_dtype(frame[preferred]):
        return preferred
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numerical columns are available for this operation.")
    return numeric_columns[0]

def finish_plot():
    """Format and display the current Matplotlib figure."""
    plt.tight_layout()
    plt.show()

def assessment_8(data):
    section("ASSESSMENT 8: K-MEANS AND HIERARCHICAL CLUSTERING")
    feature_names = [c for c in ["prestige", "educ", "babies", "preteen", "teens", "tvhours"] if c in data.columns]
    numeric_data = data[feature_names].apply(pd.to_numeric, errors="coerce").dropna()

    # 1–4. View dataset, missing values and happiness distribution.
    print("Dataset sample:\n", data.head())
    print("\nMissing values:\n", data.isna().sum())
    if "happy" in data:
        print("\nHappiness distribution:\n", data["happy"].value_counts(dropna=False))

    # 5–6. Select numerical features and standardize them.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_data)

    # 7–9. K-Means clustering with k = 2.
    kmeans2 = KMeans(n_clusters=2, n_init=25, random_state=RANDOM_STATE)
    labels2 = kmeans2.fit_predict(scaled)
    print("\nK-Means (k=2) inertia/WCSS:", kmeans2.inertia_)
    print("Cluster centers in standardized units:\n", kmeans2.cluster_centers_)
    print("Cluster sizes:\n", pd.Series(labels2).value_counts().sort_index())

    # 10. Calculate WCSS for k = 1 to 10.
    wcss = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, n_init=25, random_state=RANDOM_STATE)
        model.fit(scaled)
        wcss.append(model.inertia_)
    print("\nWCSS for k=1..10:\n", wcss)

    # 11. Elbow plot.
    plt.figure()
    plt.plot(range(1, 11), wcss, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Plot")
    finish_plot()

    # 12. Select k=3 after inspecting the elbow plot.
    final_kmeans = KMeans(n_clusters=3, n_init=25, random_state=RANDOM_STATE)
    final_labels = final_kmeans.fit_predict(scaled)
    print("\nFinal K-Means (k=3) cluster centers:\n", final_kmeans.cluster_centers_)
    print("Final cluster sizes:\n", pd.Series(final_labels).value_counts().sort_index())

    # Hierarchical clustering: use a reproducible sample because distance matrices
    # for all 17,000+ rows would consume excessive memory.
    sample_size = min(1000, len(numeric_data))
    sample = numeric_data.sample(sample_size, random_state=RANDOM_STATE)
    sample_scaled = StandardScaler().fit_transform(sample)

    # 6–7. Euclidean distance is used by Ward hierarchical clustering.
    # scipy linkage computes the hierarchical tree without explicitly storing a square matrix.
    linkage_matrix = linkage(sample_scaled, method="ward", metric="euclidean")

    # 8. Dendrogram.
    plt.figure(figsize=(12, 5))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=30, show_leaf_counts=True)
    plt.title("Hierarchical Clustering Dendrogram - Ward's Method")
    plt.xlabel("Samples or merged groups")
    plt.ylabel("Distance")
    finish_plot()

    # 9–10. Cut the tree into 3 clusters and count observations.
    hierarchical_labels = fcluster(linkage_matrix, t=3, criterion="maxclust")
    print("\nHierarchical cluster assignments (sample):\n", hierarchical_labels[:30])
    print("Number of samples in each hierarchical cluster:\n", pd.Series(hierarchical_labels).value_counts().sort_index())


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_8(data)

if __name__ == "__main__":
    main()
