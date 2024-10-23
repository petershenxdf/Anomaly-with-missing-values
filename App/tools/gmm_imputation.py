import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

def gmm_imputation(data, n_components=5, max_iter=100):
    """
    Imputes missing values using Gaussian Mixture Model Imputation.

    Parameters:
    - data: pandas DataFrame with missing values.
    - n_components: Number of mixture components.
    - max_iter: Maximum number of iterations for the EM algorithm.

    Returns:
    - Imputed pandas DataFrame.
    """
    data_imputed = data.copy()
    observed_data = data_imputed.dropna()
    missing_data = data_imputed[data_imputed.isnull().any(axis=1)]

    # Simple imputation to initialize missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed_initial = pd.DataFrame(imputer.fit_transform(data_imputed), columns=data.columns)

    # Fit GMM on the observed data
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, random_state=0)
    gmm.fit(observed_data)

    # Impute missing values
    for index, row in missing_data.iterrows():
        # Find columns with missing values
        missing_cols = row[row.isnull()].index
        observed_cols = row[row.notnull()].index

        # If all values are missing, impute with mean
        if len(observed_cols) == 0:
            data_imputed.loc[index, missing_cols] = data_imputed_initial.loc[index, missing_cols]
            continue

        # Prepare full feature vector with initial imputations
        full_row = data_imputed_initial.loc[index].values.reshape(1, -1)

        # Predict the component probabilities
        probs = gmm.predict_proba(full_row)
        # Get the most probable component
        component = np.argmax(probs, axis=1)[0]

        # Get the mean and covariance of the component
        mean = gmm.means_[component]
        cov = gmm.covariances_[component]

        # Partition the mean and covariance
        idx_observed = [data.columns.get_loc(col) for col in observed_cols]
        idx_missing = [data.columns.get_loc(col) for col in missing_cols]

        mean_observed = mean[idx_observed]
        mean_missing = mean[idx_missing]

        cov_oo = cov[np.ix_(idx_observed, idx_observed)]
        cov_mo = cov[np.ix_(idx_missing, idx_observed)]

        # Observed values
        observed_values = row[observed_cols].values.reshape(-1, 1)

        # Conditional expectation E[missing | observed]
        try:
            cov_oo_inv = np.linalg.inv(cov_oo)
        except np.linalg.LinAlgError:
            # If covariance matrix is singular, skip imputation for this row
            data_imputed.loc[index, missing_cols] = data_imputed_initial.loc[index, missing_cols]
            continue

        conditional_mean = mean_missing.reshape(-1, 1) + cov_mo @ cov_oo_inv @ (observed_values - mean_observed.reshape(-1, 1))

        # Impute missing values
        data_imputed.loc[index, missing_cols] = conditional_mean.flatten()

    return data_imputed

if __name__ == "__main__":
    # User-defined parameters
    csv_file = 'anomaly.csv'  # Replace with your CSV file name
    missing_fraction = 0.1      # Fraction of values to set as missing (e.g., 0.1 for 10%)
    n_components = 5            # Number of mixture components for GMM

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Keep a copy of the original data for evaluation
    df_original = df.copy()

    # Randomly introduce missing values
    np.random.seed(0)  # For reproducibility
    mask = np.random.rand(*df.shape) < missing_fraction
    df = df.mask(mask)

    # Impute missing values
    df_imputed = gmm_imputation(df, n_components=n_components)

    # Evaluate imputation accuracy
    mask_missing = mask
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])

    print("Evaluation Metrics for GMM Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
