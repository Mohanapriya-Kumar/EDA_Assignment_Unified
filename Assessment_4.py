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

# ASSESSMENT 4: Bivariate Analysis

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

def assessment_4(data):
    section("ASSESSMENT 4: BIVARIATE ANALYSIS")
    data = data.copy()

    # 1–2. Compare happiness level and region with contingency and stacked bar charts.
    if "happy" in data and "region" in data:
        data["Happiness_Level"] = data["happy"].map({
            "very happy": "High", "pretty happy": "Medium", "not too happy": "Low"
        }).fillna("Low")
        contingency = pd.crosstab(data["region"], data["Happiness_Level"])
        print("\nRegion by happiness contingency table:\n", contingency)

        contingency.plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.title("Happiness Level by Region - Stacked Bar Plot")
        plt.xlabel("Region")
        plt.ylabel("Frequency")
        finish_plot()

        contingency.plot(kind="bar", stacked=True, figsize=(10, 5), colormap="Set2")
        plt.title("Happiness Level by Region - Alternative Stacked Plot")
        finish_plot()

        # 3. Side-by-side/grouped bar plot.
        contingency.plot(kind="bar", stacked=False, figsize=(10, 5))
        plt.title("Happiness Levels Across Regions - Grouped Bar Plot")
        plt.xlabel("Region")
        plt.ylabel("Frequency")
        finish_plot()

    # 4. Scatter plot: education and occupational prestige.
    if {"educ", "prestige"}.issubset(data.columns):
        plt.figure()
        sns.scatterplot(data=data, x="educ", y="prestige", alpha=0.5)
        plt.title("Education and Occupational Prestige")
        plt.xlabel("Years of Education")
        plt.ylabel("Occupational Prestige")
        finish_plot()

    # 5. Trend of proportion of "very happy" responses over years.
    if {"year", "happy"}.issubset(data.columns):
        yearly = data.assign(is_very_happy=data["happy"].eq("very happy")).groupby("year", observed=False)["is_very_happy"].mean()
        plt.figure()
        plt.plot(yearly.index, yearly.values, marker="o")
        plt.title("Trend of Very Happy Responses Over Years")
        plt.xlabel("Year")
        plt.ylabel("Proportion Very Happy")
        finish_plot()

    # 6. Correlation heatmap of selected numerical variables.
    numeric_names = [c for c in ["year", "educ", "prestige", "babies", "preteen", "teens", "tvhours"] if c in data.columns]
    numeric_data = data[numeric_names].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_data.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    finish_plot()

    # 7. Compare mean education and prestige.
    avg_columns = [c for c in ["educ", "prestige"] if c in data.columns]
    if avg_columns:
        means = data[avg_columns].mean(numeric_only=True)
        plt.figure()
        means.plot(kind="bar", color=["orange", "lightgreen"][:len(means)])
        plt.title("Average Education and Prestige")
        plt.ylabel("Average Value")
        finish_plot()

    # 8. Compare TV viewing distributions across regions.
    if {"tvhours", "region"}.issubset(data.columns):
        plt.figure()
        sns.kdeplot(data=data, x="tvhours", hue="region", common_norm=False)
        plt.title("TV Hours Distribution Across Regions")
        finish_plot()

    # 9–11. Education distribution by region using box, violin, and combined plots.
    if {"educ", "region"}.issubset(data.columns):
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=data, x="region", y="educ")
        plt.title("Education Levels Between Regions")
        plt.xticks(rotation=45)
        finish_plot()

        plt.figure(figsize=(10, 5))
        sns.violinplot(data=data, x="region", y="educ", inner="quartile")
        plt.title("Education Distribution Across Regions")
        plt.xticks(rotation=45)
        finish_plot()

        plt.figure(figsize=(10, 5))
        sns.violinplot(data=data, x="region", y="educ", inner=None, color="lightblue")
        sns.boxplot(data=data, x="region", y="educ", width=0.15, color="white")
        plt.title("Violin and Box Plot: Education by Region")
        plt.xticks(rotation=45)
        finish_plot()

    # 12. Correlation matrix and pairwise correlation visualization.
    if len(numeric_names) >= 2:
        plt.figure(figsize=(9, 7))
        sns.heatmap(correlation, annot=True, cmap="vlag", center=0)
        plt.title("Correlogram of Numerical Variables")
        finish_plot()


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_4(data)

if __name__ == "__main__":
    main()
