import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error

class DensityReachableImputerMST:
    def __init__(self, n_neighbors=5):
        """
        Initialize the imputer with the number of density-reachable points (neighbors).
        
        Parameters:
        n_neighbors (int): Number of nearest density-reachable points to use for imputation.
        """
        self.n_neighbors = n_neighbors
        self.data = None
        self.cache = {}
    
    def fit(self, X):
        """
        Fit the imputer on the dataset.
        
        Parameters:
        X (array-like): Input data with missing values (NaNs).
        """
        self.data = np.array(X)
        self.n_samples, self.n_features = self.data.shape
        self.missing_mask = np.isnan(self.data)
    
    def transform(self, X):
        """
        Impute missing values in the dataset.
        
        Parameters:
        X (array-like): Input data with missing values (NaNs).
        
        Returns:
        X_imputed (array-like): Data with missing values imputed.
        """
        X = np.array(X)
        X_imputed = X.copy()
        # Process each row with missing values
        for i in range(self.n_samples):
            if self.missing_mask[i].any():
                X_imputed[i] = self._impute_row(i)
        return X_imputed
    
    def fit_transform(self, X):
        """
        Fit the imputer and transform the dataset.
        
        Parameters:
        X (array-like): Input data with missing values (NaNs).
        
        Returns:
        X_imputed (array-like): Data with missing values imputed.
        """
        self.fit(X)
        return self.transform(X)
    
    def _impute_row(self, row_index):
        """
        Impute missing values for a single row.
        
        Parameters:
        row_index (int): Index of the row to impute.
        
        Returns:
        x_imputed (array-like): The imputed row.
        """
        x = self.data[row_index]
        missing_features = self.missing_mask[row_index]
        observed_features = ~missing_features

        # Return original row if all features are missing
        if not observed_features.any():
            return x

        # Cache key based on observed features
        cache_key = tuple(observed_features)
        if cache_key in self.cache:
            mst_matrix, observed_data, valid_rows = self.cache[cache_key]
        else:
            observed_data = self.data[:, observed_features]
            valid_rows = ~np.isnan(observed_data).any(axis=1)
            # Compute pairwise distances
            diffs = observed_data[valid_rows][:, np.newaxis, :] - observed_data[valid_rows][np.newaxis, :, :]
            distances = np.sqrt(np.sum(diffs ** 2, axis=2))
            # Build MST
            mst_matrix = minimum_spanning_tree(distances)
            self.cache[cache_key] = (mst_matrix, observed_data, valid_rows)

        if not valid_rows.any():
            return x  # No valid rows to compare

        x_imputed = x.copy()
        # Get indices of valid rows
        valid_indices = np.where(valid_rows)[0]
        # Map the row_index to the index in valid_indices
        try:
            root_index_in_valid = np.where(valid_indices == row_index)[0][0]
        except IndexError:
            # The root is not in valid_rows (missing observed features), cannot impute
            return x

        # Traverse MST starting from root to get n nearest points
        n_reachable_indices = self._get_n_reachable_points(mst_matrix, root_index_in_valid, self.n_neighbors)
        neighbor_rows = valid_indices[n_reachable_indices]

        for feature_idx in np.where(missing_features)[0]:
            neighbor_values = self.data[neighbor_rows, feature_idx]
            neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
            if neighbor_values.size > 0:
                x_imputed[feature_idx] = neighbor_values.mean()
        return x_imputed

        return x_imputed

    def _get_n_reachable_points(self, mst_matrix, root_index, n):
        """
        Get n earliest expanded points from the MST starting from the root index.

        Parameters:
        mst_matrix (csr_matrix): MST represented as a sparse matrix.
        root_index (int): Index of the root node in the MST.
        n (int): Number of points to retrieve.

        Returns:
        reachable_indices (list): Indices of the n reachable points.
        """
        from collections import deque

        n_nodes = mst_matrix.shape[0]
        visited = [False] * n_nodes
        reachable_indices = []

        adjacency_list = [set() for _ in range(n_nodes)]
        coo = mst_matrix.tocoo()
        for i, j in zip(coo.row, coo.col):
            adjacency_list[i].add(j)
            adjacency_list[j].add(i)  # Since MST is undirected

        queue = deque()
        queue.append(root_index)
        visited[root_index] = True

        while queue and len(reachable_indices) < n:
            current = queue.popleft()
            if current != root_index:
                reachable_indices.append(current)
            for neighbor in adjacency_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return reachable_indices[:n]

def evaluate_imputation_accuracy(file_path, missing_rate=0.1, n_neighbors=5, random_state=42):
    """
    Evaluate the imputation accuracy of the DensityReachableImputerMST on a given dataset.

    Parameters:
    - file_path (str): Path to the CSV file.
    - missing_rate (float): Fraction of data to be set as missing (between 0 and 1).
    - n_neighbors (int): Number of density-reachable points to use for imputation.
    - random_state (int): Seed for random number generator.

    Returns:
    - metrics (dict): Dictionary containing MAE and RMSE for each feature and overall.
    """
    np.random.seed(random_state)
    
    # Load the dataset
    data = pd.read_csv(file_path)
    data = data.select_dtypes(include=[np.number])  # Use only numeric columns for simplicity
    data_values = data.values

    # Store the original data for comparison
    original_data = data_values.copy()

    # Introduce missing values
    n_samples, n_features = data_values.shape
    total_values = n_samples * n_features
    n_missing = int(np.floor(missing_rate * total_values))

    # Randomly choose indices to set as missing
    missing_indices = np.unravel_index(
        np.random.choice(total_values, n_missing, replace=False),
        (n_samples, n_features)
    )
    data_values[missing_indices] = np.nan

    # Initialize the imputer
    imputer = DensityReachableImputerMST(n_neighbors=n_neighbors)

    # Perform imputation
    imputed_data = imputer.fit_transform(data_values)

    # Compute accuracy metrics
    # Only compute error on the introduced missing values
    mask = np.isnan(data_values)
    mae_per_feature = []
    rmse_per_feature = []
    all_true_values = []
    all_imputed_values = []

    for i in range(n_features):
        feature_mask = mask[:, i]
        if feature_mask.any():
            true_values = original_data[feature_mask, i]
            imputed_values = imputed_data[feature_mask, i]
            all_true_values.extend(true_values)
            all_imputed_values.extend(imputed_values)
            mae = mean_absolute_error(true_values, imputed_values)
            rmse = np.sqrt(mean_squared_error(true_values, imputed_values))
            mae_per_feature.append(mae)
            rmse_per_feature.append(rmse)
        else:
            mae_per_feature.append(np.nan)
            rmse_per_feature.append(np.nan)

    # Compute overall MAE and RMSE
    overall_mae = mean_absolute_error(all_true_values, all_imputed_values)
    overall_rmse = np.sqrt(mean_squared_error(all_true_values, all_imputed_values))

    metrics = {
        'MAE_per_feature': mae_per_feature,
        'RMSE_per_feature': rmse_per_feature,
        'Overall_MAE': overall_mae,
        'Overall_RMSE': overall_rmse
    }

    # Print the results
    feature_names = data.columns.tolist()
    print("Imputation Accuracy Metrics:")
    for i, feature in enumerate(feature_names):
        mae_i = mae_per_feature[i]
        rmse_i = rmse_per_feature[i]
        if not np.isnan(mae_i):
            print(f"Feature '{feature}': MAE = {mae_i:.4f}, RMSE = {rmse_i:.4f}")
        else:
            print(f"Feature '{feature}': No missing values introduced.")

    print(f"\nOverall MAE: {overall_mae:.4f}")
    print(f"Overall RMSE: {overall_rmse:.4f}")

    return metrics


# Example usage:
if __name__ == "__main__":
    # Set parameters
    file_path = 'anomaly.csv'  # Path to your CSV file
    missing_rate = 0.1         # 10% of the data will be set as missing
    n_neighbors = 5            # Number of neighbors for imputation

    # Evaluate imputation accuracy
    evaluate_imputation_accuracy(file_path, missing_rate, n_neighbors)
