from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

def apply_imputation(df):
    # Define 5 different imputation methods
    imputers = [
        SimpleImputer(strategy='mean'),
        SimpleImputer(strategy='median'),
        SimpleImputer(strategy='most_frequent'),
        SimpleImputer(strategy='constant', fill_value=0),  # Replace with zero
        KNNImputer(n_neighbors=3)
    ]
    
    imputed_datasets = []
    for imputer in imputers:
        imputed = imputer.fit_transform(df)
        imputed_datasets.append(imputed)

    averaged_data = np.mean(imputed_datasets, axis=0)  # Take average of imputations

    # Store each set of imputed points separately
    imputation_points = np.array(imputed_datasets)

    return averaged_data, imputation_points.tolist()
