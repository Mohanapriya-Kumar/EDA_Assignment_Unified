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

# ASSESSMENT 7: 1D, 2D and ND Descriptive Analysis

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

def assessment_7(data):
    section("ASSESSMENT 7: 1D, 2D AND ND ANALYSIS")
    data = data.copy()

    # 2–3. Summary statistics for prestige.
    x = data["prestige"].dropna()
    stats = {
        "Mean": x.mean(),
        "Median": x.median(),
        "Minimum": x.min(),
        "Maximum": x.max(),
        "Range": x.max() - x.min(),
        "Variance": x.var(),
        "Standard Deviation": x.std(),
        "IQR": x.quantile(0.75) - x.quantile(0.25),
        "Skewness": skew(x, bias=False),
        "Kurtosis": kurtosis(x, fisher=False, bias=False)
    }
    summary_stats = pd.DataFrame({"Statistic": list(stats.keys()), "Value": list(stats.values())})
    print("\nSummary statistics:\n", summary_stats)

    # 4. Frequency and relative frequency with pie and bar plots.
    freq = data["happy"].value_counts(dropna=False)
    rel_freq = data["happy"].value_counts(normalize=True, dropna=False)
    print("\nFrequency:\n", freq)
    print("\nRelative frequency:\n", rel_freq)
    plt.figure(figsize=(7, 7))
    plt.pie(freq.values, labels=freq.index.astype(str), autopct="%1.1f%%")
    plt.title("Happiness Distribution")
    finish_plot()
    plt.figure(figsize=(8, 5))
    plt.bar(freq.index.astype(str), freq.values)
    plt.title("Happiness Distribution")
    plt.xlabel("Happiness")
    plt.ylabel("Frequency")
    plt.xticks(rotation=15)
    finish_plot()

    # 5. Binned frequency distribution, histogram and box plot.
    bins = pd.cut(data["prestige"], bins=5)
    print("\nPrestige bin frequencies:\n", bins.value_counts().sort_index())
    plt.figure()
    sns.histplot(data=data, x="prestige", bins=5)
    plt.title("Prestige Distribution")
    finish_plot()
    plt.figure()
    sns.boxplot(y=data["prestige"])
    plt.title("Prestige Box Plot")
    finish_plot()

    # 6–7. Contingency table, row/column proportions and stacked bar plot.
    tab = pd.crosstab(data["region"], data["happy"])
    print("\nContingency table:\n", tab)
    print("\nRow percentages:\n", tab.div(tab.sum(axis=1), axis=0))
    print("\nColumn percentages:\n", tab.div(tab.sum(axis=0), axis=1))
    tab.plot(kind="bar", stacked=True, figsize=(10, 5), title="Region and Happiness")
    plt.ylabel("Frequency")
    finish_plot()

    # 8–9. Grouped statistics, side-by-side box plot and violin plot.
    grouped = data.groupby("happy", observed=False)["prestige"].agg(["mean", "median", "std"])
    print("\nPrestige summary by happiness:\n", grouped)
    plt.figure()
    sns.boxplot(data=data, x="happy", y="prestige")
    plt.title("Prestige by Happiness")
    plt.xticks(rotation=20)
    finish_plot()
    plt.figure()
    sns.violinplot(data=data, x="happy", y="prestige")
    plt.title("Prestige Distribution by Happiness")
    plt.xticks(rotation=20)
    finish_plot()

    # 10. Pearson and Spearman correlations plus regression trend line.
    pair = data[["prestige", "educ"]].dropna()
    print("\nPearson correlation:", pearsonr(pair["prestige"], pair["educ"]))
    print("Spearman correlation:", spearmanr(pair["prestige"], pair["educ"]))
    plt.figure()
    sns.regplot(data=pair, x="prestige", y="educ", scatter_kws={"alpha": 0.4})
    plt.title("Prestige vs Education")
    finish_plot()

    # 11. Summary statistics for combinations of region and happiness.
    summary_nd = data.groupby(["region", "happy"], observed=False)["prestige"].agg(["mean", "std"])
    print("\nPrestige mean and SD by region and happiness:\n", summary_nd)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data, x="happy", y="prestige")
    plt.title("Prestige by Happiness, Faceted by Region")
    plt.xticks(rotation=20)
    finish_plot()

    # 12. Scatter plot by category and category-wise correlation.
    plt.figure()
    sns.scatterplot(data=data, x="prestige", y="educ", hue="happy", style="happy", alpha=0.5)
    plt.title("Prestige vs Education by Happiness")
    finish_plot()
    for category, group in data.groupby("happy", observed=False):
        group_pair = group[["prestige", "educ"]].dropna()
        if len(group_pair) >= 2:
            print(f"Correlation for {category}: {group_pair.corr().iloc[0, 1]:.4f}")

    # 13–14. Correlation matrix heatmap and pairs plot.
    num_cols = [c for c in ["prestige", "educ", "babies", "preteen", "teens", "tvhours"] if c in data.columns]
    corr_matrix = data[num_cols].corr()
    print("\nCorrelation matrix:\n", corr_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix Heatmap")
    finish_plot()
    pair_data = data[num_cols[:4]].dropna()
    sns.pairplot(pair_data.sample(min(1000, len(pair_data)), random_state=RANDOM_STATE))
    plt.suptitle("Pairs Plot", y=1.02)
    plt.show()


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_7(data)

if __name__ == "__main__":
    main()
