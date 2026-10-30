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

# ASSESSMENT 1: Dataframe Operations and Data Transformations

import warnings
import math
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
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def finish_plot():
    plt.tight_layout()
    plt.show()

def assessment_1():
    section("ASSESSMENT 1: MERGING, DATA TRANSFORMATION AND NORMALIZATION")

    # dataframe1 - Pantry essentials
    dataframe1 = pd.DataFrame({
        "ProductID": range(101, 111),
        "NameofProd": [
            "Rice", "Toor Dal", "Gingelly Oil", "Chilly Powder", "Coffee Powder",
            "Tea", "Gram Dal", "Asafoetida", "Jaggery", "Urad Dal"
        ],
        "Price": [320, 160, 240, 65, 180, 55, 40, 50, 90, 290]
    })
    print("\nDataframe 1 - Pantry essentials:\n", dataframe1)

    # dataframe2 - ProductId and company location
    dataframe2 = pd.DataFrame({
        "ProductID": range(101, 111),
        "Company_Location": [
            "Thanjavur", "Erode", "Tuticorin", "Kashmir", "Kumbakonam",
            "Nilgiris", "Madurai", "Kanchipuram", "Salem", "Erode"
        ]
    })
    print("\nDataframe 2 - Product location:\n", dataframe2)

    # dataframe3 - Additional stock
    dataframe3 = pd.DataFrame({
        "ProductID": range(111, 121),
        "NameofProd": [
            "Cardamom", "Ghee", "Nendran Chips", "Dry Chilli", "Appalam",
            "Palm Jaggery", "Fenugreek", "Turmeric Powder", "Rava", "Vermicelli"
        ],
        "Price": [250, 340, 90, 30, 45, 190, 60, 55, 60, 40]
    })
    print("\nDataframe 3 - Additional stock:\n", dataframe3)

    # Append dataframe1 and dataframe3 (R rbind -> pandas concat by rows).
    new_stock = pd.concat([dataframe1, dataframe3], ignore_index=True)
    print("\nNew Stock - appended rows:\n", new_stock)

    # Concatenate side by side (R cbind -> pandas concat by columns).
    concatenated = pd.concat(
        [dataframe1.reset_index(drop=True), dataframe2[["Company_Location"]].reset_index(drop=True)],
        axis=1
    )
    print("\nConcatenated side-by-side dataframes:\n", concatenated)

    # Merge the two dataframes using ProductID.
    merged = pd.merge(dataframe1, dataframe2, on="ProductID", how="inner")
    print("\nMerged products with source:\n", merged)

    # Join operations: inner, left, right, and full outer.
    for join_type in ["inner", "left", "right", "outer"]:
        result = pd.merge(dataframe1, dataframe2, on="ProductID", how=join_type)
        print(f"\n{join_type.title()} Join:\n", result)

    # II. Data transformations using the ten-value vector.
    # Keep 0108 as the integer 108; leading zeroes are not retained by numeric types.
    fname = np.array([108, 45, 120, 15, 88, 510, 62, 22, 95, 70], dtype=float)
    data_min, data_max = fname.min(), fname.max()
    data_mean, data_sd = fname.mean(), fname.std(ddof=1)

    # Normalize to [-3, 3] and [-1, 1] using min-max scaling.
    range_3 = ((fname - data_min) / (data_max - data_min)) * 6 - 3
    range_1 = ((fname - data_min) / (data_max - data_min)) * 2 - 1
    min_max_scaled = (fname - data_min) / (data_max - data_min)

    # Z-score transformation uses sample standard deviation, matching R's sd().
    z_scores = (fname - data_mean) / data_sd

    # Decimal scaling: divide by 10 raised to the number of digits in max absolute value.
    scale_power = math.ceil(math.log10(np.max(np.abs(fname)) + 1e-12))
    decimal_scaled = fname / (10 ** scale_power)

    print("\nOriginal vector:", fname)
    print("Scaled to [-3, 3]:", range_3)
    print("Scaled to [-1, 1]:", range_1)
    print("Min-max scaled [0, 1]:", min_max_scaled)
    print("Z-scores:", z_scores)
    print("Decimal-scaled values:", decimal_scaled)


def main():
    assessment_1()

if __name__ == "__main__":
    main()
