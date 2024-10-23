#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 22:31:01 2024

@author: jiahao
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def kde_imputation(data, bandwidth=1.0):
    """
    Imputes missing values using Kernel Density Estimation.

    Parameters:
    - data: pandas DataFrame with missing values.
    - bandwidth: Bandwidth parameter for KDE.

    Returns:
    - Imputed pandas DataFrame.
    """
    data_imputed = data.copy()
    for column in data.columns:
        if data[column].isnull().any():
            # Get non-missing values for the column
            observed_values = data[column].dropna().values.reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(observed_values)

            # Number of missing values
            n_missing = data[column].isnull().sum()

            # Sample from the estimated density
            imputed_values = kde.sample(n_missing).flatten()

            # Fill in the missing values
            data_imputed.loc[data_imputed[column].isnull(), column] = imputed_values
    return data_imputed

if __name__ == "__main__":
    # User-defined parameters
    csv_file = 'anomaly.csv'  # Replace with your CSV file name
    missing_fraction = 0.1      # Fraction of values to set as missing (e.g., 0.1 for 10%)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Keep a copy of the original data for evaluation
    df_original = df.copy()

    # Randomly introduce missing values
    np.random.seed(0)  # For reproducibility
    mask = np.random.rand(*df.shape) < missing_fraction
    df[mask] = np.nan

    # Impute missing values
    df_imputed = kde_imputation(df, bandwidth=1.0)

    # Evaluate imputation accuracy
    mask_missing = mask
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])

    print("Evaluation Metrics for KDE Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
