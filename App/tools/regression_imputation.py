import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_imputation(data):
    """
    Imputes missing values using regression imputation.

    Parameters:
    - data: pandas DataFrame with missing values.

    Returns:
    - Imputed pandas DataFrame.
    """
    data_imputed = data.copy()
    for column in data.columns:
        if data[column].isnull().any():
            # Use other columns as predictors
            df_train = data_imputed[data_imputed[column].notnull()]
            df_missing = data_imputed[data_imputed[column].isnull()]

            X_train = df_train.drop(columns=[column])
            y_train = df_train[column]
            X_missing = df_missing.drop(columns=[column])

            # Simple imputation for missing values in predictors
            imputer = SimpleImputer(strategy='mean')
            X_train_imputed = imputer.fit_transform(X_train)
            X_missing_imputed = imputer.transform(X_missing)

            # Train regression model
            model = LinearRegression()
            model.fit(X_train_imputed, y_train)

            # Predict missing values
            y_pred = model.predict(X_missing_imputed)

            # Fill in the missing values
            data_imputed.loc[data_imputed[column].isnull(), column] = y_pred
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
    df_imputed = regression_imputation(df)

    # Evaluate imputation accuracy
    mask_missing = mask
    mse = mean_squared_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    mae = mean_absolute_error(df_original.values[mask_missing], df_imputed.values[mask_missing])
    r2 = r2_score(df_original.values[mask_missing], df_imputed.values[mask_missing])

    print("Evaluation Metrics for Regression Imputation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
