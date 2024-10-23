import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def svd_imputation(data, n_components=5):
    """
    Imputes missing values using Matrix Factorization (SVD).

    Parameters:
    - data: pandas DataFrame with missing values.
    - n_components: Number of singular values to keep.

    Returns:
    - Imputed pandas DataFrame.
    """
    # Simple imputation to fill missing values temporarily
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)

    # Perform SVD
    svd = TruncatedSVD(n_components=n_components)
    U = svd.fit_transform(data_imputed)
    Sigma = svd.singular_values_
    Vt = svd.components_

    # Reconstruct the data matrix
    data_reconstructed = np.dot(U, np.dot(np.diag(Sigma), Vt))

    # Put back into DataFrame
    data_imputed = pd.DataFrame(data_reconstructed, columns=data.columns)

    # Only replace the missing values
    data_imputed = data.copy().where(~data.isnull(), data_imputed)

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
    df_imputed = svd_imputation(df, n_components=5)

    # Evaluate imputation accuracy
    mask_missing = mask
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])

    print("Evaluation Metrics for SVD Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
