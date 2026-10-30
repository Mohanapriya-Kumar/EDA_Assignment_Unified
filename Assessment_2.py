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

# ASSESSMENT 2: Data Cleaning and Missing Values

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

def assessment_2(data):
    section("ASSESSMENT 2: DATA IMPORT, TRANSFORMATION AND BASIC STATISTICS")

    # 1–3. Display first and last ten records.
    print("\nFirst 10 records:\n", data.head(10))
    print("\nLast 10 records:\n", data.tail(10))

    # 4–7. Dimensions, column names, data types, and summary statistics.
    print("\nNumber of rows and columns:", data.shape)
    print("Number of rows:", len(data))
    print("Number of columns:", len(data.columns))
    print("Column names:\n", data.columns.tolist())
    print("\nData types and non-missing counts:")
    data.info()
    print("\nSummary statistics:\n", data.describe(include="all").T)

    # 8–10. Missing-value counts by column and for the entire dataset.
    print("\nMissing values in each column:\n", data.isna().sum())
    print("Total missing values:", int(data.isna().sum().sum()))

    # 11. Identify rows containing at least one missing value.
    rows_with_missing = data[data.isna().any(axis=1)]
    print("\nRows with at least one missing value:\n", rows_with_missing.head(10))

    # 12. Remove rows with missing values.
    clean_data = data.dropna()
    print("\nCleaned dataset after dropping missing rows:\n", clean_data.head(10))
    print("Cleaned shape:", clean_data.shape)

    # 13–15. Demonstrate mean, median, and mode imputation on a separate copy.
    imputed_data = data.copy()
    if "prestige" in imputed_data:
        imputed_data["prestige"] = imputed_data["prestige"].fillna(imputed_data["prestige"].mean())
    if "tvhours" in imputed_data:
        imputed_data["tvhours"] = imputed_data["tvhours"].fillna(imputed_data["tvhours"].median())
    if "income" in imputed_data:
        imputed_data["income"] = imputed_data["income"].fillna(mode_value(imputed_data["income"]))

    # 16–17. Forward fill and backward fill.
    forward_fill = data.ffill()
    backward_fill = data.bfill()
    print("\nForward-filled sample:\n", forward_fill.head(10))
    print("\nBackward-filled sample:\n", backward_fill.head(10))

    # 18. Replace remaining missing values with 0 for numeric and "Unknown" for text.
    constant_fill = data.copy()
    for column in constant_fill.columns:
        if pd.api.types.is_numeric_dtype(constant_fill[column]):
            constant_fill[column] = constant_fill[column].fillna(0)
        else:
            constant_fill[column] = constant_fill[column].fillna("Unknown")
    print("\nConstant-filled sample:\n", constant_fill.head(10))

    # 19. Remove duplicate records.
    unique_data = data.drop_duplicates()
    print("\nRows after duplicate removal:", len(unique_data))

    # 20. Convert appropriate columns to categorical data types (R factors).
    for column in ["workstat", "income", "region", "happy"]:
        if column in imputed_data.columns:
            imputed_data[column] = imputed_data[column].astype("category")
    print("\nData types after categorical conversion:\n", imputed_data.dtypes)

    # 21. Rename one column.
    renamed_data = imputed_data.rename(columns={"workstat": "employment_status"})
    print("\nRenamed columns:\n", renamed_data.columns.tolist())

    # 22. Create a derived column by adding child age-group counts.
    if all(c in renamed_data.columns for c in ["babies", "preteen", "teens"]):
        renamed_data["total_children"] = renamed_data[["babies", "preteen", "teens"]].sum(axis=1, min_count=1)
        print("\nDerived total_children column:\n", renamed_data[["babies", "preteen", "teens", "total_children"]].head(10))

    # 23–28. Summary and descriptive statistics for prestige.
    if "prestige" in data:
        x = data["prestige"].dropna()
        print("\nSummary statistics:\n", x.describe())
        print("Minimum:", x.min())
        print("Maximum:", x.max())
        print("Range:", x.max() - x.min())
        print("Variance:", x.var())
        print("Standard deviation:", x.std())
        print("Interquartile range:", x.quantile(0.75) - x.quantile(0.25))


def main():
    try:
        data = load_happiness_data()
    except Exception as exc:
        print("Could not load happiness.csv. Check your internet connection.")
        print("Error:", exc)
        return
    print("Loaded happiness dataset shape:", data.shape)
    assessment_2(data)

if __name__ == "__main__":
    main()
