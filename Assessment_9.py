"""
Fall Semester 2026-2027
Digital Assignment 
Name: Mohanapriya Kumar
Reg. No: 23BDS0108
Course Code: BCSE331L
Course Name: Exploratory Data Analysis 
Course Type: Th
Class Number: VL2026270102447
Slot: A2
"""

# ASSESSMENT 9: Principal Component Analysis (PCA)

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

def assessment_9(data):
    section("ASSESSMENT 9: PCA USING THE HAPPINESS DATASET")

    # 1–2. Display the dataset dimensions.
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])

    # 3. Select numeric columns, clean missing/infinite values, and remove constants.
    numeric = data.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).dropna()
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.shape[1] < 2:
        print("At least two varying numerical columns are required for PCA.")
        return

    # Standardize the variables so each variable has mean 0 and standard deviation 1.
    data_scaled = StandardScaler().fit_transform(numeric)

    # 4. Perform PCA and summarize variance explained by each component.
    pca = PCA()
    pca_scores_array = pca.fit_transform(data_scaled)
    variance_percent = pca.explained_variance_ratio_ * 100
    pca_summary = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(len(variance_percent))],
        "Standard Deviation": np.sqrt(pca.explained_variance_),
        "Proportion of Variance": pca.explained_variance_ratio_,
        "Cumulative Proportion": np.cumsum(pca.explained_variance_ratio_)
    })
    print("\nPCA summary:\n", pca_summary)

    # 5. Plot percentage of variance explained by each component.
    # The y-axis is limited to 0–60 to match the assessment requirement.
    components = np.arange(1, len(variance_percent) + 1)
    plt.figure(figsize=(12, 5))
    plt.bar(components, variance_percent, color="steelblue")
    for component, value in zip(components, variance_percent):
        if value >= 1:
            plt.text(component, value + 0.5, f"{value:.2f}%", ha="center", fontsize=7)
    plt.ylim(0, 60)
    plt.xlabel("Principal Components")
    plt.ylabel("Percentage of Variance Explained")
    plt.title("Scree Plot of Principal Components")
    finish_plot()

    # 6. Create a biplot-style visualization of observations and variable loadings.
    pca_scores = pd.DataFrame(
        pca_scores_array,
        columns=[f"PC{i + 1}" for i in range(pca_scores_array.shape[1])],
        index=numeric.index
    )
    loadings = pd.DataFrame(
        pca.components_.T,
        index=numeric.columns,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)]
    )

    plt.figure(figsize=(10, 7))
    plt.scatter(pca_scores["PC1"], pca_scores["PC2"],
                color="steelblue", alpha=0.35, s=12)
    scale_factor = max(pca_scores[["PC1", "PC2"]].abs().max()) * 0.65
    for variable in numeric.columns:
        x_loading = loadings.loc[variable, "PC1"]
        y_loading = loadings.loc[variable, "PC2"]
        plt.arrow(0, 0, x_loading * scale_factor, y_loading * scale_factor,
                  color="red", alpha=0.65, head_width=0.04, length_includes_head=True)
        plt.text(x_loading * scale_factor * 1.08,
                 y_loading * scale_factor * 1.08,
                 variable, fontsize=7, color="darkred")
    plt.axhline(0, color="gray", linewidth=0.7)
    plt.axvline(0, color="gray", linewidth=0.7)
    plt.title("PCA Biplot - Happiness Dataset")
    plt.xlabel(f"PC1 ({variance_percent[0]:.2f}% variance)")
    plt.ylabel(f"PC2 ({variance_percent[1]:.2f}% variance)")
    finish_plot()

    # 7. Extract PCA scores and display the first six rows.
    print("\nFirst six PCA score rows:\n", pca_scores.head(6))

    # 8. Plot PC1 versus PC2 using blue circular points and the registration number.
    plt.figure()
    plt.scatter(pca_scores["PC1"], pca_scores["PC2"],
                color="blue", marker="o", s=25)
    plt.title("PCA Scatter Plot - Reg. No.: 23BDS0108")
    plt.xlabel("First Principal Component (PC1)")
    plt.ylabel("Second Principal Component (PC2)")
    finish_plot()


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_9(data)

if __name__ == "__main__":
    main()
