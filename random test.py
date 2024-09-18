#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:26:15 2024

@author: jiahao
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from cuml.metrics import pairwise_distances
from sklearn.manifold import MDS
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

# Set default renderer to 'browser' to ensure it opens in the web browser
pio.renderers.default = 'browser'

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for the dataset
num_points = 500  # Total number of data points (adjusted for clarity in visualization)
num_features = 10  # Number of attributes/features
missing_value_row_percentage = 0.1  # Percentage of rows with missing values

# Generate synthetic data
data = np.random.randn(num_points, num_features)

# Introduce missing values in only 10% of the rows
num_missing_rows = int(num_points * missing_value_row_percentage)
missing_rows = np.random.choice(num_points, num_missing_rows, replace=False)

# Set multiple attributes to NaN in selected rows
for row in missing_rows:
    num_missing_in_row = np.random.randint(1, num_features)  # At least one missing value per row
    missing_cols = np.random.choice(num_features, num_missing_in_row, replace=False)
    data[row, missing_cols] = np.nan

# Convert to DataFrame
df = pd.DataFrame(data, columns=[f'Feature_{i+1}' for i in range(num_features)])

# Only select rows that have missing values for imputation
rows_with_missing_values = df.isnull().any(axis=1)
data_with_missing_values = df[rows_with_missing_values]

# Impute only rows that originally had missing values
# Mean Imputation
mean_imputer = SimpleImputer(strategy='mean')
data_mean_imputed = mean_imputer.fit_transform(data_with_missing_values)

# Median Imputation
median_imputer = SimpleImputer(strategy='median')
data_median_imputed = median_imputer.fit_transform(data_with_missing_values)

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
data_knn_imputed = knn_imputer.fit_transform(data_with_missing_values)

# Combine the original complete data with the imputed data
original_complete_data = df.dropna().values
data_combined = np.vstack([
    original_complete_data, 
    data_mean_imputed, 
    data_median_imputed, 
    data_knn_imputed
])

# Compute pairwise distances using GPU acceleration
distance_matrix = pairwise_distances(data_combined)

# Use MDS to reduce to 2D
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_results = mds.fit_transform(distance_matrix)

# Prepare the labels and group IDs for interactive plot
num_original = len(original_complete_data)
num_imputed = len(data_mean_imputed)

labels = (['Original'] * num_original +
          ['Mean Imputed'] * num_imputed +
          ['Median Imputed'] * num_imputed +
          ['KNN Imputed'] * num_imputed)

# Assign group numbers to track the imputed points
group_numbers = list(range(1, num_original + 1)) + list(range(1, num_imputed + 1)) * 3

# Creating a DataFrame for Plotly
mds_df = pd.DataFrame({
    'MDS1': mds_results[:, 0],
    'MDS2': mds_results[:, 1],
    'Label': labels,
    'Group': group_numbers
})

# Plotting with Plotly
fig = px.scatter(
    mds_df, 
    x='MDS1', 
    y='MDS2', 
    color='Label',
    symbol='Label',
    hover_data=['Label', 'Group'],
    title='Interactive MDS Visualization of Original and Imputed Data Points'
)

# Update layout to improve interactivity
fig.update_layout(
    legend=dict(title='Data Points'),
    xaxis_title='MDS Dimension 1',
    yaxis_title='MDS Dimension 2',
    template='plotly'
)

# Show the plot
fig.show(renderer="browser")

# Optional: Save to an HTML file for manual viewing
fig.write_html("interactive_mds_plot.html")

