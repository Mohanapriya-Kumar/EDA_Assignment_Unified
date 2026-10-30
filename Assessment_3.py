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

# ASSESSMENT 3: Univariate Analysis

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

def assessment_3(data):
    section("ASSESSMENT 3: DATA CLEANING AND UNIVARIATE ANALYSIS")
    data = data.copy()

    # 1–4. Import is handled by the shared loader; inspect records and missing values.
    print("First six records:\n", data.head(6))
    print("Last six records:\n", data.tail(6))
    print("Total missing values:", int(data.isna().sum().sum()))
    print("Rows containing missing values:\n", data[data.isna().any(axis=1)].head(10))

    # 5. Remove columns with more than 70% missing values.
    missing_fraction = data.isna().mean()
    data = data.loc[:, missing_fraction <= 0.70].copy()

    # 6–7. Create mean-imputed and median-imputed versions of numeric columns.
    mean_data = data.copy()
    median_data = data.copy()
    numeric_columns = data.select_dtypes(include=np.number).columns
    for column in numeric_columns:
        mean_data[column] = mean_data[column].fillna(mean_data[column].mean())
        median_data[column] = median_data[column].fillna(median_data[column].median())
    print("\nMean-imputed sample:\n", mean_data.head())
    print("\nMedian-imputed sample:\n", median_data.head())

    # Continue with median-imputed data, as in the R program.
    data = median_data

    # 8. Replace missing categorical values with each column's mode.
    for column in data.select_dtypes(include=["object", "category"]).columns:
        data[column] = data[column].fillna(mode_value(data[column]))

    # 9–10. Identify and remove duplicate rows.
    duplicates = data[data.duplicated(keep=False)]
    print("\nDuplicate rows:\n", duplicates.head(10))
    data = data.drop_duplicates()

    # 11–13. Inspect data types and standardize text to lowercase.
    print("\nData types:\n", data.dtypes)
    for column in data.select_dtypes(include="object").columns:
        data[column] = data[column].astype("category")
    for column in data.select_dtypes(include="category").columns:
        data[column] = data[column].astype(str).str.lower().astype("category")
    print("\nData types after categorical conversion:\n", data.dtypes)

    # Select prestige as the example numeric variable, matching the R experiment.
    numeric_column = first_numeric_column(data, "prestige")
    values = data[numeric_column].dropna()

    # 14. Detect outliers using a boxplot.
    plt.figure()
    sns.boxplot(y=values)
    plt.title(f"Boxplot of {numeric_column}")
    plt.ylabel(numeric_column)
    finish_plot()

    # 15–16. Identify outliers with IQR and remove extreme outliers.
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = data[(data[numeric_column] < lower_bound) | (data[numeric_column] > upper_bound)]
    print(f"\nIQR outliers for {numeric_column}:\n", outliers[[numeric_column]].head(20))
    data_no_outliers = data[data[numeric_column].between(lower_bound, upper_bound) | data[numeric_column].isna()]
    print("Shape before/after removing IQR outliers:", data.shape, data_no_outliers.shape)

    # 17–18. Histogram and density curve.
    plt.figure()
    sns.histplot(data[numeric_column], kde=False, bins=30)
    plt.title(f"Histogram of {numeric_column}")
    plt.xlabel(numeric_column)
    plt.ylabel("Frequency")
    finish_plot()

    plt.figure()
    sns.histplot(data[numeric_column], kde=True, stat="density", bins=30)
    plt.title(f"Histogram with Density Curve: {numeric_column}")
    finish_plot()

    # 19. Violin plot of the selected numerical variable.
    plt.figure()
    sns.violinplot(y=data[numeric_column])
    plt.title(f"Violin Plot of {numeric_column}")
    finish_plot()

    # 20. Dot plot for a small numerical sample.
    small_values = data[numeric_column].dropna().head(30).reset_index(drop=True)
    plt.figure()
    plt.scatter(small_values.index, small_values.values)
    plt.title(f"Dot Plot of First 30 {numeric_column} Values")
    plt.xlabel("Observation")
    plt.ylabel(numeric_column)
    finish_plot()

    # 21–22. Frequency, proportion, and percentage tables for happiness.
    categorical_column = "happy" if "happy" in data.columns else data.select_dtypes(include=["category", "object"]).columns[0]
    frequency = data[categorical_column].value_counts(dropna=False)
    proportion = data[categorical_column].value_counts(normalize=True, dropna=False)
    print("\nFrequency table:\n", frequency)
    print("\nProportion table:\n", proportion)
    print("\nPercentage table:\n", proportion * 100)

    # 23–25. Vertical, horizontal, and percentage bar charts.
    plt.figure()
    frequency.plot(kind="bar")
    plt.title(f"Frequency of {categorical_column}")
    plt.xlabel(categorical_column)
    plt.ylabel("Frequency")
    finish_plot()

    plt.figure()
    frequency.sort_values().plot(kind="barh")
    plt.title(f"Horizontal Frequency Chart of {categorical_column}")
    plt.xlabel("Frequency")
    finish_plot()

    plt.figure()
    (proportion * 100).plot(kind="bar")
    plt.title(f"Percentage Distribution of {categorical_column}")
    plt.ylabel("Percentage")
    finish_plot()

    # 26. Pie chart for categories.
    plt.figure()
    frequency.plot(kind="pie", autopct="%1.1f%%")
    plt.title(f"Pie Chart of {categorical_column}")
    plt.ylabel("")
    finish_plot()

    # 27–28. Lollipop chart and dot chart of categorical frequencies.
    ordered = frequency.sort_values()
    plt.figure()
    plt.hlines(y=ordered.index.astype(str), xmin=0, xmax=ordered.values)
    plt.plot(ordered.values, ordered.index.astype(str), "o")
    plt.title(f"Lollipop Chart of {categorical_column}")
    plt.xlabel("Frequency")
    finish_plot()

    plt.figure()
    plt.plot(ordered.values, ordered.index.astype(str), "o")
    plt.title(f"Dot Chart of {categorical_column}")
    plt.xlabel("Frequency")
    finish_plot()

    # 29. Density plot.
    plt.figure()
    sns.kdeplot(data=data, x=numeric_column, fill=True)
    plt.title(f"Density Plot of {numeric_column}")
    finish_plot()

    # 30. Faceted distributions across groups, if both columns exist.
    if "region" in data.columns:
        plot_data = data[[numeric_column, "region"]].dropna()
        g = sns.displot(data=plot_data, x=numeric_column, col="region", col_wrap=3, kind="hist")
        g.fig.suptitle(f"{numeric_column} Distribution by Region", y=1.02)
        plt.show()


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_3(data)

if __name__ == "__main__":
    main()
