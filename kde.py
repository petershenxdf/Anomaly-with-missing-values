# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# KDE Imputation Function
def kde_impute(data, missing_value=np.nan, bandwidth='auto'):
    """
    Impute missing values using Kernel Density Estimation (KDE).

    Parameters:
    data (np.array): 1D numpy array where missing values are represented by `missing_value`
    missing_value (float or int): Value representing the missing data (default is np.nan)
    bandwidth (float or 'auto'): Bandwidth for the KDE. If 'auto', cross-validation is used to select the best bandwidth.

    Returns:
    np.array: Data with missing values imputed.
    """

    # Separate observed (non-missing) data
    observed_data = data[~np.isnan(data)]
    missing_data_mask = np.isnan(data)

    # Estimate optimal bandwidth if not provided
    if bandwidth == 'auto':
        bandwidths = np.logspace(-1, 1, 20)  # Range of bandwidths to try
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=5)  # Cross-validation to find best bandwidth
        grid.fit(observed_data.reshape(-1, 1))
        kde = grid.best_estimator_
    else:
        kde = KernelDensity(bandwidth=bandwidth)
        kde.fit(observed_data.reshape(-1, 1))

    # Generate samples for missing values based on the estimated density
    n_missing = missing_data_mask.sum()
    imputed_values = kde.sample(n_samples=n_missing).flatten()

    # Impute missing values with sampled values
    imputed_data = data.copy()
    imputed_data[missing_data_mask] = imputed_values

    return imputed_data

# Function to introduce missing values and evaluate imputation
def test_kde_imputation_on_real_data(data, missing_percentage):
    """
    Function to test KDE imputation on real data with a specified percentage of missing values.

    Parameters:
    data (pd.DataFrame): Real dataset
    missing_percentage (float): Percentage of data to make missing

    Returns:
    float: Mean Squared Error of imputed values vs original values
    """
    # Create a copy of the data and introduce missing values
    data_with_missing = data.copy()
    missing_mask = np.random.rand(*data_with_missing.shape) < missing_percentage
    data_with_missing[missing_mask] = np.nan

    # Apply KDE Imputation on the dataset
    imputed_data = data_with_missing.apply(lambda x: kde_impute(x.values, bandwidth='auto') if x.isnull().sum() > 0 else x)

    # Calculate accuracy (mean squared error) between imputed and true values for missing points
    true_values = data[missing_mask]
    imputed_values = imputed_data[missing_mask]
    mse = mean_squared_error(true_values, imputed_values)
    print(true_values)
    print(imputed_values)
    # Print and return the MSE
    print(f"Mean Squared Error (MSE) for {missing_percentage * 100:.1f}% missing data: {mse:.4f}")
    
    # Plot original, observed, and imputed data for visualization
    plt.figure(figsize=(10, 6))
    plt.hist(data_with_missing.dropna().values.flatten(), bins=30, alpha=0.5, label='Observed Data')
    plt.hist(imputed_values, bins=30, alpha=0.5, label='Imputed Data')
    plt.hist(data.values.flatten(), bins=30, alpha=0.3, label='True Data')
    plt.legend()
    plt.title(f"KDE Imputation - True vs Imputed Data\nMean Squared Error (MSE): {mse:.4f}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

    return mse

# Load the real dataset (CSV file)
def load_and_test_real_dataset(file_path, missing_percentage):
    """
    Load the real dataset from a CSV file and test KDE imputation.

    Parameters:
    file_path (str): Path to the CSV file containing the real dataset
    missing_percentage (float): Percentage of data to make missing
    """
    # Load the dataset
    data = pd.read_csv(file_path)

    # Run the test with the specified missing percentage
    test_kde_imputation_on_real_data(data, missing_percentage)

# Example usage
file_path = 'anomaly.csv'  # Replace with the correct file path if needed
missing_percentage = 0.2  # Specify the percentage of missing data
load_and_test_real_dataset(file_path, missing_percentage)
