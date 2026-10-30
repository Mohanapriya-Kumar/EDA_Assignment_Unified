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

# ASSESSMENT 5: Multivariate Analysis

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

def assessment_5(data):
    section("ASSESSMENT 5: MULTIVARIATE ANALYSIS")
    data = data.copy()

    # 1. Examine data and identify numeric/categorical variables.
    print("\nData types:\n", data.dtypes)
    print("\nSummary:\n", data.describe(include="all").T)
    num_vars = [c for c in ["year", "prestige", "educ", "babies", "preteen", "teens", "tvhours"] if c in data.columns]
    cat_vars = [c for c in ["workstat", "region", "happy", "attend", "divorce", "owngun"] if c in data.columns]
    print("Numerical variables:", num_vars)
    print("Categorical variables:", cat_vars)

    # 2. Scatter plot with a third variable represented by color.
    plt.figure()
    sns.scatterplot(data=data, x="prestige", y="educ", hue="happy", alpha=0.6)
    plt.title("Prestige vs Education by Happiness")
    finish_plot()

    # 3. Scatter plot with color and shape.
    plt.figure()
    sns.scatterplot(data=data, x="prestige", y="educ", hue="happy", style="divorce", alpha=0.6)
    plt.title("Prestige vs Education by Happiness and Divorce")
    finish_plot()

    # 4. Scatter plot with color and point size.
    plt.figure()
    sns.scatterplot(data=data, x="prestige", y="educ", hue="happy", size="tvhours", sizes=(15, 150), alpha=0.6)
    plt.title("Prestige vs Education by Happiness and TV Hours")
    finish_plot()

    # 5. Bubble plot: bubble size represents number of babies.
    plt.figure()
    sns.scatterplot(data=data, x="prestige", y="educ", size="babies", sizes=(15, 250), alpha=0.5, legend=True)
    plt.title("Bubble Plot of Prestige, Education and Babies")
    finish_plot()

    # 6. Facet-wrap equivalent: separate panels by happiness.
    g = sns.relplot(data=data, x="prestige", y="educ", col="happy", col_wrap=3, kind="scatter", alpha=0.5)
    g.fig.suptitle("Prestige vs Education by Happiness", y=1.03)
    plt.show()

    # 7. Facet-grid equivalent: rows are regions and columns are happiness categories.
    # Limit to categories with enough data so the grid remains readable.
    subset = data.dropna(subset=["region", "happy", "prestige", "educ"])
    top_regions = subset["region"].value_counts().head(5).index
    subset = subset[subset["region"].isin(top_regions)]
    g = sns.relplot(data=subset, x="prestige", y="educ", row="region", col="happy", kind="scatter", alpha=0.4)
    g.fig.suptitle("Prestige vs Education by Region and Happiness", y=1.02)
    plt.show()

    # 8. Pairs plot for four numerical variables.
    pair_columns = [c for c in ["prestige", "educ", "babies", "preteen"] if c in data.columns]
    pair_data = data[pair_columns].dropna()
    sns.pairplot(pair_data.sample(min(1000, len(pair_data)), random_state=RANDOM_STATE))
    plt.suptitle("Pairs Plot", y=1.02)
    plt.show()

    # 9. Parallel coordinate plot: normalize selected variables to [0, 1].
    parallel_columns = pair_columns
    if len(parallel_columns) >= 2 and "region" in data.columns:
        parallel = data[parallel_columns + ["region"]].dropna().copy()
        parallel = parallel.sample(min(300, len(parallel)), random_state=RANDOM_STATE)
        scaled = MinMaxScaler().fit_transform(parallel[parallel_columns])
        scaled_df = pd.DataFrame(scaled, columns=parallel_columns)
        scaled_df["region"] = parallel["region"].astype(str).to_numpy()
        plt.figure(figsize=(10, 5))
        for region, group in scaled_df.groupby("region"):
            for _, row in group.iterrows():
                plt.plot(parallel_columns, row[parallel_columns].values, alpha=0.08)
        plt.title("Parallel Coordinate Plot (scaled values)")
        plt.xlabel("Variables")
        plt.ylabel("Scaled Values")
        finish_plot()


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_5(data)

if __name__ == "__main__":
    main()
