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
            
            # Scale the observed data
            scaler = StandardScaler()
            observed_data_scaled = scaler.fit_transform(observed_data)
            
            # Calculate bandwidth using Silverman's Rule of Thumb
            n = len(observed_data_scaled)
            sigma = np.std(observed_data_scaled, ddof=1)
            bandwidth = (4 * (sigma ** 5) / (3 * n)) ** (1 / 5)
            bandwidth = max(bandwidth, 1e-3)  # Ensure bandwidth is not too small
            
            # Fit KDE to observed data
            kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
            kde.fit(observed_data_scaled)
            
            # Generate a grid over which to evaluate the KDE
            x_min, x_max = observed_data_scaled.min(), observed_data_scaled.max()
            x_grid = np.linspace(x_min - 3 * bandwidth, x_max + 3 * bandwidth, 1000).reshape(-1, 1)
            #print(x_grid)
            # Evaluate the KDE on the grid
            log_density = kde.score_samples(x_grid)
            density = np.exp(log_density)
            
            # Compute the expected value (mean) over the grid
            mean_value_scaled = np.sum(x_grid.flatten() * density) / np.sum(density)
            
            # Inverse transform to get back to original scale
            mean_value = scaler.inverse_transform([[mean_value_scaled]])[0, 0]
            
            data_imputed.loc[missing_indices, column] = mean_value
    return data_imputed

# Function to get dataset statistics for evaluation
def get_dataset_statistics(data):
    stats = {
        'mean': data.mean(),
        'median': data.median(),
        'std_dev': data.std(),
        'min': data.min(),
        'max': data.max(),
        'skewness': data.skew(),
        'kurtosis': data.kurt()
    }
    return stats

# Function to evaluate imputation accuracy per column
def evaluate_imputation_accuracy_per_column(original_data, imputed_data, mask_missing):
    mae_per_feature = []
    rmse_per_feature = []
    feature_names = original_data.columns.tolist()
    
    for i, column in enumerate(feature_names):
        mask = mask_missing[:, i]
        if mask.any():
            true_values = original_data.iloc[:, i].values[mask]
            imputed_values = imputed_data.iloc[:, i].values[mask]
            mae = mean_absolute_error(true_values, imputed_values)
            rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
            mae_per_feature.append(mae)
            rmse_per_feature.append(rmse)
        else:
            mae_per_feature.append(np.nan)
            rmse_per_feature.append(np.nan)
    
    # Print the results
    print("Imputation Accuracy Metrics:")
    for i, feature in enumerate(feature_names):
        mae_i = mae_per_feature[i]
        rmse_i = rmse_per_feature[i]
        if not np.isnan(mae_i):
            print(f"Feature '{feature}': MAE = {mae_i:.4f}, RMSE = {rmse_i:.4f}")
        else:
            print(f"Feature '{feature}': No missing values introduced.")
    
    # Compute overall metrics
    mask_flat = mask_missing.flatten()
    true_values_all = original_data.values.flatten()[mask_flat]
    imputed_values_all = imputed_data.values.flatten()[mask_flat]
    overall_mae = mean_absolute_error(true_values_all, imputed_values_all)
    overall_rmse = np.sqrt(mean_squared_error(true_values_all, imputed_values_all))
    print(f"\nOverall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")
    
    metrics = {
        'MAE_per_feature': mae_per_feature,
        'RMSE_per_feature': rmse_per_feature,
        'Overall_MAE': overall_mae,
        'Overall_RMSE': overall_rmse
    }
    return metrics

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('anomaly.csv')
    
    # Get and print dataset statistics before introducing missing values
    stats_original = get_dataset_statistics(df)
    print("Original Dataset Statistics:")
    for key, value in stats_original.items():
        print(f"{key.capitalize()}:\n{value}\n")
    
    # Keep a copy of the original data for evaluation
    df_original = df.copy()
    
    # Introduce missing values (10% of the cells)
    missing_fraction = 0.1
    df_missing, mask_missing = introduce_missing_values(df, missing_fraction=missing_fraction)
    
    # Impute missing values using KDE with Silverman's Rule of Thumb for bandwidth
    df_imputed = kde_imputation(df_missing)
    
    # Evaluate imputation accuracy per column
    metrics = evaluate_imputation_accuracy_per_column(df_original, df_imputed, mask_missing)
    
    # Get and print dataset statistics after imputation
    stats_imputed = get_dataset_statistics(df_imputed)
    print("\nImputed Dataset Statistics:")
    for key, value in stats_imputed.items():
        print(f"{key.capitalize()}:\n{value}\n")
