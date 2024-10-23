import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def knn_imputation(data, n_neighbors=5):
    """
    Imputes missing values using K-Nearest Neighbors imputation.

    Parameters:
    - data: pandas DataFrame with missing values.
    - n_neighbors: Number of neighboring samples to use.

    Returns:
    - Imputed pandas DataFrame.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    data_imputed = imputer.fit_transform(data)
    return pd.DataFrame(data_imputed, columns=data.columns)

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
    df_imputed = knn_imputation(df, n_neighbors=5)

    # Evaluate imputation accuracy
    mask_missing = mask
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])

    print("Evaluation Metrics for KNN Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
