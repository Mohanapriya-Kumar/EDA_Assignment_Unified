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

# ASSESSMENT 6: Time Series Analysis - Air Passengers

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
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def finish_plot():
    plt.tight_layout()
    plt.show()

def assessment_6():
    section("ASSESSMENT 6: TIME SERIES ANALYSIS – AIR PASSENGERS")

    # R's built-in AirPassengers dataset is represented here as monthly values.
    # It contains 144 monthly observations from January 1949 through December 1960.
    passenger_values = [
        112,118,132,129,121,135,148,148,136,119,104,118,
        115,126,141,135,125,149,170,170,158,133,114,140,
        145,150,178,163,172,178,199,199,184,162,146,166,
        171,180,193,181,183,218,230,242,209,191,172,194,
        196,196,236,235,229,243,264,272,237,211,180,201,
        204,188,235,227,234,264,302,293,259,229,203,229,
        242,233,267,269,270,315,364,347,312,274,237,278,
        284,277,317,313,318,374,413,405,355,306,271,306,
        315,301,356,348,355,422,465,467,404,347,305,336,
        340,318,362,348,363,435,491,505,404,359,310,337,
        360,342,406,396,420,472,548,559,463,407,362,405,
        417,391,419,461,472,535,622,606,508,461,390,432
    ]
    dates = pd.date_range("1949-01-01", periods=len(passenger_values), freq="MS")
    air_data = pd.Series(passenger_values, index=dates, name="Passengers")

    # 1–2. Load and inspect first and last observations.
    print("First five observations:\n", air_data.head())
    print("\nLast five observations:\n", air_data.tail())

    # 3–4. Starting/ending periods, frequency, and time-series characteristics.
    print("\nStart:", air_data.index.min())
    print("End:", air_data.index.max())
    print("Frequency: monthly (12 observations per year)")
    print("Python object type:", type(air_data))

    # 5. Visualize monthly passenger trend.
    plt.figure(figsize=(10, 5))
    plt.plot(air_data.index, air_data.values)
    plt.title("Monthly Air Passengers")
    plt.xlabel("Year")
    plt.ylabel("Number of Passengers")
    finish_plot()

    # 6–7. Decompose the series into trend, seasonal, and residual components.
    decomposition = seasonal_decompose(air_data, model="multiplicative", period=12)
    decomposition.plot()
    plt.suptitle("AirPassengers: Trend, Seasonal and Residual Components")
    finish_plot()

    # 8. Autocorrelation and partial autocorrelation.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    plot_acf(air_data, lags=36, ax=axes[0], title="Autocorrelation of AirPassengers")
    plot_pacf(air_data, lags=36, ax=axes[1], method="ywm", title="Partial Autocorrelation")
    finish_plot()

    # 9–10. Fit an ARIMA(1, 1, 1) model.
    model = ARIMA(air_data, order=(1, 1, 1))
    fitted_model = model.fit()
    print("\nARIMA(1,1,1) model summary:\n", fitted_model.summary())

    # 11–12. Forecast the next 12 months and plot actual plus forecast.
    forecast = fitted_model.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    confidence = forecast.conf_int()
    print("\n12-month forecast:\n", forecast_mean)

    plt.figure(figsize=(11, 5))
    plt.plot(air_data.index, air_data.values, label="Actual")
    plt.plot(forecast_mean.index, forecast_mean.values, label="Forecast", color="blue")
    plt.fill_between(confidence.index, confidence.iloc[:, 0], confidence.iloc[:, 1], alpha=0.2)
    plt.title("AirPassengers Forecast")
    plt.xlabel("Year")
    plt.ylabel("Number of Passengers")
    plt.legend()
    finish_plot()


def main():
    assessment_6()

if __name__ == "__main__":
    main()
