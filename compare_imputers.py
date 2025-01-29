# compare_imputers.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import KNNImputer

# Import MST-based imputer
from density_imputer import DensityReachableImputerMST

##############################################################################
#                         Synthetic Data Generators                          #
##############################################################################

def generate_arc_data(n_points=100, random_state=42):
    """
    Generate a synthetic dataset of two arcs in 2D.
    Returns an array of shape (n_points, 2).
    """
    np.random.seed(random_state)
    half = n_points // 2
    
    # Arc 1
    angles1 = np.linspace(0, np.pi, half)
    x1 = np.cos(angles1)
    y1 = np.sin(angles1)
    
    # Arc 2 (shifted to [2,0])
    angles2 = np.linspace(0, np.pi, half)
    x2 = 2 + np.cos(angles2)
    y2 = np.sin(angles2)
    
    X = np.vstack((np.column_stack((x1, y1)),
                   np.column_stack((x2, y2))))
    return X

def generate_spiral_data(n_points=100, turns=2, noise=0.1, random_state=42):
    """
    Generate a spiral dataset in 2D, which can be non-convex. 
    Returns array of shape (n_points, 2).
    """
    np.random.seed(random_state)
    t = np.linspace(0, turns * 2 * np.pi, n_points)
    x = t * np.cos(t)
    y = t * np.sin(t)
    # Add some noise
    x += np.random.normal(scale=noise, size=n_points)
    y += np.random.normal(scale=noise, size=n_points)
    return np.column_stack((x, y))

def introduce_missingness(X, missing_rate=0.2, random_state=42):
    """
    Randomly introduce missing values into X at the specified rate.
    """
    np.random.seed(random_state)
    X_missing = X.copy()
    n_samples, n_features = X_missing.shape
    n_missing = int(np.floor(missing_rate * n_samples * n_features))
    missing_indices = np.random.choice(n_samples * n_features, n_missing, replace=False)
    rows = missing_indices // n_features
    cols = missing_indices % n_features
    X_missing[rows, cols] = np.nan
    return X_missing

##############################################################################
#                      Comparison & Visualization                            #
##############################################################################

def compare_imputers(X_full, missing_rate=0.3, n_neighbors=5, title=""):
    """
    Given a dataset X_full (no missing values), artificially introduce missingness,
    then compare MST-based imputation with KNN-based imputation.
    
    Plots the original data, MST-imputed data, and KNN-imputed data side by side.
    Prints and returns the imputation errors (MAE, RMSE).
    """
    # Introduce missingness
    X_missing = introduce_missingness(X_full, missing_rate=missing_rate)

    # Keep track of where the missing entries were introduced
    mask_missing = np.isnan(X_missing)

    # 1) Impute with MST approach
    mst_imputer = DensityReachableImputerMST(n_neighbors=n_neighbors)
    X_mst_imputed = mst_imputer.fit_transform(X_missing)

    # 2) Impute with KNNImputer
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    X_knn_imputed = knn_imputer.fit_transform(X_missing)

    # Evaluate errors only on the introduced missing positions
    true_values = X_full[mask_missing]
    mst_values  = X_mst_imputed[mask_missing]
    knn_values  = X_knn_imputed[mask_missing]

    # ------------------------------------------------------------------
    # Filter out any NaNs that remain post-imputation in mst_values/knn_values
    # so that sklearn metrics won't raise an error.
    # ------------------------------------------------------------------
    valid_mst = ~np.isnan(mst_values)
    valid_knn = ~np.isnan(knn_values)

    # MST Imputer metrics
    if np.any(valid_mst):
        mae_mst  = mean_absolute_error(true_values[valid_mst], mst_values[valid_mst])
        rmse_mst = np.sqrt(mean_squared_error(true_values[valid_mst], mst_values[valid_mst]))
    else:
        # If no valid entries were imputed (everything is still NaN),
        # we cannot compute a proper metric. We'll set them to NaN.
        mae_mst  = float('nan')
        rmse_mst = float('nan')

    # KNN Imputer metrics
    if np.any(valid_knn):
        mae_knn  = mean_absolute_error(true_values[valid_knn], knn_values[valid_knn])
        rmse_knn = np.sqrt(mean_squared_error(true_values[valid_knn], knn_values[valid_knn]))
    else:
        mae_knn  = float('nan')
        rmse_knn = float('nan')

    # Print out results
    print(f"=== {title} ===")
    print(f"MST Imputer: MAE = {mae_mst:.4f},  RMSE = {rmse_mst:.4f}")
    print(f"KNN Imputer: MAE = {mae_knn:.4f},  RMSE = {rmse_knn:.4f}\n")

    # ------------------------------- Plot -------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=14)

    # Original data
    axes[0].scatter(X_full[:, 0], X_full[:, 1],
                    c='blue', alpha=0.6, s=30)
    axes[0].set_title("Original (No Missing)")

    # MST-imputed data
    axes[1].scatter(X_mst_imputed[:, 0], X_mst_imputed[:, 1],
                    c=['red' if m else 'blue' for m in mask_missing.any(axis=1)],
                    alpha=0.6, s=30)
    axes[1].set_title("MST-Imputed")

    # KNN-imputed data
    axes[2].scatter(X_knn_imputed[:, 0], X_knn_imputed[:, 1],
                    c=['red' if m else 'blue' for m in mask_missing.any(axis=1)],
                    alpha=0.6, s=30)
    axes[2].set_title("KNN-Imputed")

    for ax in axes:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
    plt.tight_layout()
    plt.show()

    return {
        "MST_MAE": mae_mst,
        "MST_RMSE": rmse_mst,
        "KNN_MAE": mae_knn,
        "KNN_RMSE": rmse_knn
    }

if __name__ == "__main__":
    # 1) Arcs dataset
    X_arcs = generate_arc_data(n_points=100, random_state=0)
    compare_imputers(X_arcs, missing_rate=0.3, n_neighbors=5, title="Two Arcs")

    # 2) Spiral dataset
    X_spiral = generate_spiral_data(n_points=100, turns=2, noise=0.15, random_state=0)
    compare_imputers(X_spiral, missing_rate=0.3, n_neighbors=5, title="Spiral Data")
