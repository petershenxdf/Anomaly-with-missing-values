import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to introduce missing values into the dataset
def introduce_missing_values(data, missing_fraction=0.1, random_state=0):
    np.random.seed(random_state)
    data_missing = data.copy()
    mask = np.random.rand(*data.shape) < missing_fraction
    data_missing[mask] = np.nan
    return data_missing, mask

# KDE-based imputation function
def kde_imputation(data):
    data_imputed = data.copy()
    for column in data.columns:
        missing_indices = data[column].isna()
        if missing_indices.any():
            # Extract observed data
            observed_data = data.loc[~missing_indices, column].values.reshape(-1, 1)
            
            # Ensure sufficient number of observed points
            if len(observed_data) < 5:
                data_imputed.loc[missing_indices, column] = np.mean(observed_data)
                continue
            
            # Scale the observed data to reduce variance
            scaler = StandardScaler()
            observed_data_scaled = scaler.fit_transform(observed_data)
            
            # Calculate bandwidth using Silverman's Rule of Thumb
            n = len(observed_data)
            sigma = np.std(observed_data_scaled)
            bandwidth = (4 * (sigma ** 5) / (3 * n)) ** (1 / 5)
            
            # Fit KDE to observed data
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(observed_data_scaled)
            
            # Estimate the mean of the KDE distribution to impute missing values
            mean_value_scaled = np.mean(observed_data_scaled)
            mean_value = scaler.inverse_transform([[mean_value_scaled]])[0, 0]
            data_imputed.loc[missing_indices, column] = mean_value
    return data_imputed

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('anomaly.csv')
    
    # Keep a copy of the original data for evaluation
    df_original = df.copy()
    
    # Introduce missing values (10% of the cells)
    missing_fraction = 0.1
    df_missing, mask_missing = introduce_missing_values(df, missing_fraction=missing_fraction)
    
    # Impute missing values using KDE with Silverman's Rule of Thumb for bandwidth
    df_imputed = kde_imputation(df_missing)
    
    # Evaluate imputation accuracy
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])
    
    # Print evaluation metrics
    print("Evaluation Metrics for KDE Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
