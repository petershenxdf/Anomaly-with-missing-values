#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 16:26:11 2024

@author: jiahao
"""
import os
import pandas as pd
import cuml.metrics as cuml_metrics
from sklearn.manifold import MDS
import cupy as cp  # CuPy for GPU array management
import traceback
from flask import Flask, render_template, jsonify, request
import numpy as np
from tools.imputation_methods import apply_imputation  # Import imputation methods

app = Flask(__name__)

DATA_FOLDER = os.path.join(os.getcwd(), 'data')
TEMP_FOLDER = os.path.join(os.getcwd(), 'temp')

# Ensure the temp folder exists
if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

# Helper function to get all datasets (CSV files) in the 'data' folder
def get_datasets():
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    return files

# Helper function to save the pairwise distances to a CSV file
def save_pairwise_distances(pairwise_distances_np, dataset_name):
    temp_file_path = os.path.join(TEMP_FOLDER, f"{dataset_name}_pairwise_distances.csv")
    np.savetxt(temp_file_path, pairwise_distances_np, delimiter=',')
    return temp_file_path

def compute_mds_with_cuml_distances(data, dataset_name):
    # If the data is already a NumPy array, no need to convert
    if isinstance(data, pd.DataFrame):
        data_np = data.to_numpy()  # Convert pandas DataFrame to NumPy array
    else:
        data_np = data  # It's already a NumPy array

    # Compute pairwise distances using cuML (already returns a NumPy array)
    pairwise_distances_np = cuml_metrics.pairwise_distances(data_np)
    print(f"Pairwise distances shape: {pairwise_distances_np.shape}")
    # Save the pairwise distances to the temp folder
    save_pairwise_distances(pairwise_distances_np, dataset_name)

    # Run MDS on the precomputed pairwise distances
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_result = mds.fit_transform(pairwise_distances_np)
    return mds.fit_transform(pairwise_distances_np)
    print(f"MDS projection result shape: {mds_result.shape}")
    return mds_result 
@app.route('/')
def index():
    datasets = get_datasets()  # Get the list of datasets for the dropdown
    return render_template('dashboard.html')

@app.route('/scatterplot')
def scatterplot():
    datasets = get_datasets() 
    return render_template('scatterplot.html', datasets=datasets)

@app.route('/scatterplot-dataset', methods=['POST'])
def load_dataset():
    try:
        dataset_name = request.json['dataset']
        filepath = os.path.join(DATA_FOLDER, dataset_name)

        # Load the dataset
        df = pd.read_csv(filepath)
        print(f"Loaded dataset shape: {df.shape}")
        
        # Check if 500 rows are loaded
        #if df.shape[0] != 500:
            #print(f"Warning: Expected 500 rows but got {df.shape[0]} rows.")

        numeric_cols = df.select_dtypes(include=[np.number])
        #print(f"Numeric columns shape: {numeric_cols.shape}")

        if numeric_cols.empty:
            return jsonify({'error': 'No numeric data in the selected dataset'}), 400

         
        points=[]
        
        
        # Log rows with missing values
        rows_with_missing_values = df.isnull().any(axis=1).sum()
        print(f"Rows with missing values: {rows_with_missing_values}")
        num_points=len(df)
        all_points=np.empty((num_points+rows_with_missing_values*5,df.shape[1]))

        # Loop over the dataset rows
        for idx, row in df.iterrows():
            #print()
            current_key=len(points)
            if row.isnull().any():  # If there's a missing value
                
                imputed_points1 = np.random.randn(5, df.shape[1])# Fake imputation logic (replace with actual imputation)
                
                
                avg_imputed_point1 = imputed_points1.mean(axis=0)
                #print('avg shape:'+str(avg_imputed_point1.shape))
                
                all_points[current_key]=avg_imputed_point1
                all_points[current_key+1:current_key+6]=imputed_points1
                
                points.append({
                        'isAverage': True,
                         'isImputated':True,
                         'imputedIndices': list(range(current_key+1,current_key+6)),
                    })
                for key in range(current_key+1,current_key+6):
                    points.append({
                        'isAverage': False,
                        'isImputated':True,
                         'imputedIndices':[]
                        })
                    
               
            else:
                #non_imputed_points.append(row.values[:2])
                points.append({
                    'isAverage': False,
                    'isImputated':False,
                     'imputedIndices':[]
                    })
                all_points[current_key]=row


        #print(f"Non-imputed points: {len(non_imputed_points)}")
        #print(f"Average imputed points: {len(avg_imputed_points)}")
        #print(f"Imputed points: {len(imputed_data)}")
        #print(f"num zeros in all points: {all_num_zero}")
        #print(points)
        # Combine all points: non-imputed, average imputed points, and imputed points
        #all_points1 = np.concatenate([non_imputed_points, avg_imputed_points, imputed_data], axis=0)
        #print(f"Total points passed to MDS: {all_points.shape[0]}")

        # Compute MDS projection with distances
        mds_result = compute_mds_with_cuml_distances(all_points, dataset_name)
        print(f"MDS result shape: {mds_result.shape}")

        return jsonify({
            'points': mds_result.tolist(),
            
             'all_points':points# Send imputation data to front-end
        })

    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing the dataset'}), 500



if __name__ == '__main__':
    app.run(debug=True)
