#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 21:01:30 2024

@author: jiahao
"""
import pandas as pd

# Import the imputation functions from their respective files
from mean_median_imputation import mean_median_imputation
from knn_imputation import knn_imputation
from regression_imputation import regression_imputation
from hot_deck_imputation import hot_deck_imputation
from matrix_factorization_imputation import matrix_factorization_imputation

# Load your dataset
data = pd.read_csv('anomaly.csv')

# Specify the row index you want to impute
row_index_to_impute = 5  # Adjust this index based on your dataset

# Mean/Median Imputation
imputed_row_mean = mean_median_imputation(
    row_index=row_index_to_impute,
    data=data,
    strategy='mean'  # or 'median'
)
print("Imputed Row using Mean/Median Imputation:")
print(imputed_row_mean)

# KNN Imputation
imputed_row_knn = knn_imputation(
    row_index=row_index_to_impute,
    data=data,
    n_neighbors=5  # Adjust k as needed
)
print("\nImputed Row using KNN Imputation:")
print(imputed_row_knn)

# Regression Imputation
imputed_row_regression = regression_imputation(
    row_index=row_index_to_impute,
    data=data
)
print("\nImputed Row using Regression Imputation:")
print(imputed_row_regression)

# Hot Deck Imputation
imputed_row_hot_deck = hot_deck_imputation(
    row_index=row_index_to_impute,
    data=data
)
print("\nImputed Row using Hot Deck Imputation:")
print(imputed_row_hot_deck)

# Matrix Factorization Imputation
imputed_row_matrix = matrix_factorization_imputation(
    row_index=row_index_to_impute,
    data=data,
    rank=2  # Adjust rank as needed
)
print("\nImputed Row using Matrix Factorization Imputation:")
print(imputed_row_matrix)
