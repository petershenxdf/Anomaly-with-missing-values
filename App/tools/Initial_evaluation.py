import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the imputation functions from their respective files
from mean_median_imputation import mean_median_imputation
from knn_imputation import knn_imputation
from regression_imputation import regression_imputation
from hot_deck_imputation import hot_deck_imputation
from matrix_factorization_imputation import matrix_factorization_imputation

def introduce_missingness(data, missing_rate, random_state=None):
    """
    Randomly introduce missing values into the dataset based on the missing rate, ensuring no column is entirely missing.
    """
    np.random.seed(random_state)
    data_missing = data.copy()
    n_total = data.size
    n_missing = int(np.floor(n_total * missing_rate))
    
    print(f"Total values in the dataset: {n_total}")
    print(f"Number of missing values to introduce: {n_missing}")
    
    # Ensure that we don't remove all data from any column
    max_missing_per_column = data.shape[0] - 1  # Leave at least one value per column
    
    # Get all indices in the data
    all_indices = [(i, j) for i in range(data.shape[0]) for j in range(data.shape[1])]
    np.random.shuffle(all_indices)
    
    missing_indices = []
    missing_counts_per_column = {col: 0 for col in range(data.shape[1])}
    
    for idx in all_indices:
        if len(missing_indices) >= n_missing:
            break
        i, j = idx
        if missing_counts_per_column[j] < max_missing_per_column:
            missing_indices.append(idx)
            missing_counts_per_column[j] += 1
    
    # Set the missing values
    for i, j in missing_indices:
        data_missing.iat[i, j] = np.nan
    
    # Create a mask of missing values
    mask = data_missing.isnull()
    
    # Print the actual number of missing values per column
    missing_per_column = data_missing.isnull().sum()
    print("Missing values per column after introduction:")
    print(missing_per_column)
    
    return data_missing, mask

def evaluate_imputation_methods(data_complete, data_missing, mask):
    """
    Apply imputation methods and evaluate their performance.

    Parameters:
    - data_complete (pd.DataFrame): The original complete dataset.
    - data_missing (pd.DataFrame): The dataset with missing values.
    - mask (pd.DataFrame): Boolean mask indicating missing positions.

    Returns:
    - results (pd.DataFrame): DataFrame containing evaluation metrics for each method.
    """
    methods = {
        'Mean/Median Imputation': mean_median_imputation,
        'KNN Imputation': knn_imputation,
        'Regression Imputation': regression_imputation,
        'Hot Deck Imputation': hot_deck_imputation,
        'Matrix Factorization Imputation': matrix_factorization_imputation
    }

    metrics = {
        'MAE': mean_absolute_error,
        'MSE': mean_squared_error,
        'RMSE': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2 Score': r2_score
    }

    results = []

    for method_name, impute_function in methods.items():
        print(f"\nRunning {method_name}...")

        if method_name in ['Regression Imputation', 'Hot Deck Imputation']:
            # These methods are row-wise
            imputed_data = data_missing.copy()
            for row_index in data_missing[mask.any(axis=1)].index:
                try:
                    imputed_row = impute_function(row_index=row_index, data=data_missing)
                    imputed_data.iloc[row_index] = imputed_row
                except Exception as e:
                    print(f"Error in {method_name} for row {row_index}: {e}")
                    continue
        else:
            # Methods that impute the entire DataFrame
            try:
                if method_name == 'Mean/Median Imputation':
                    imputed_data = impute_function(data_missing, strategy='mean')
                else:
                    imputed_data = impute_function(data_missing)
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                continue

        # Check for remaining NaNs
        remaining_nans = imputed_data[mask].isnull().sum().sum()
        print(f"Number of missing values remaining after {method_name}: {remaining_nans}")

        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} missing values remain after {method_name}.")
            # Exclude positions with remaining NaNs from evaluation
            mask_valid = mask & ~imputed_data.isnull()
        else:
            mask_valid = mask

        # Flatten the arrays for evaluation
        y_true = data_complete[mask_valid].values.flatten()
        y_pred = imputed_data[mask_valid].values.flatten()

        # Ensure there are valid values to evaluate
        if len(y_true) == 0:
            print(f"No valid imputed values for {method_name}. Skipping evaluation.")
            continue

        method_results = {'Method': method_name}
        for metric_name, metric_function in metrics.items():
            score = metric_function(y_true, y_pred)
            method_results[metric_name] = score

        results.append(method_results)

    results_df = pd.DataFrame(results)
    return results_df

def main():
    # Load a complete dataset
    iris = load_iris()
    data_complete = pd.DataFrame(iris.data, columns=iris.feature_names)

    # User-specified missing rate
    missing_rate = 0.1  # 10% of the data will be set to missing

    # Introduce missingness
    data_missing, mask = introduce_missingness(data_complete, missing_rate, random_state=42)

    # Evaluate imputation methods
    results_df = evaluate_imputation_methods(data_complete, data_missing, mask)

    # Display the results
    print("\nImputation Method Evaluation Results:")
    print(results_df)

if __name__ == "__main__":
    main()
