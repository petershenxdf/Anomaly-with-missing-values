import numpy as np
import pandas as pd

def hot_deck_imputation(row_index, data):
    """
    Imputes missing values in the specified row using hot deck imputation.

    Parameters:
    - row_index (int): Index of the row to impute.
    - data (pd.DataFrame): The entire dataset.

    Returns:
    - imputed_row (np.ndarray): The imputed row as a NumPy array.
    """
    # Extract the row to impute
    row = data.iloc[row_index].copy()

    # Identify missing and observed columns in the row
    missing_cols = row[row.isnull()].index.tolist()
    observed_cols = row[row.notnull()].index.tolist()

    # Ensure there are observed values to match donors
    if not observed_cols:
        # Fall back to mean imputation
        row[missing_cols] = data[missing_cols].mean()
        return row.values

    # Remove the row to impute from the dataset
    donor_pool = data.drop(index=row_index)

    # Remove donors with missing values in observed columns
    donor_pool = donor_pool.dropna(subset=observed_cols)

    if donor_pool.empty:
        # Fall back to mean imputation
        row[missing_cols] = data[missing_cols].mean()
        return row.values

    # Calculate distances between the row and donors based on observed columns
    distances = donor_pool[observed_cols].apply(
        lambda x: np.linalg.norm(x - row[observed_cols]), axis=1
    )

    # Find the closest donor
    closest_donor_index = distances.idxmin()
    donor_row = donor_pool.loc[closest_donor_index]

    # Impute missing values with donor's values
    for col in missing_cols:
        row[col] = donor_row[col]

    # Return the imputed row as a NumPy array
    return row.values
