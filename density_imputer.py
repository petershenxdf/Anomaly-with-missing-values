#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:59:20 2025

@author: jiahao
"""
# density_imputer.py
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

class DensityReachableImputerMST:
    """
    Impute missing values by traversing a Minimum Spanning Tree among
    'density-reachable' rows. For each row that needs imputation:
     - Identify which features are observed vs. missing
     - Build or retrieve from cache an MST of valid rows (for the observed features)
     - BFS from the row-of-interest along MST edges to find up to k reachable points
     - Impute each missing feature using the average of those k neighbors
    """
    def __init__(self, n_neighbors=5):
        """
        Parameters:
        -----------
        n_neighbors: int
            Number of MST neighbors to use when imputing missing features
        """
        self.n_neighbors = n_neighbors
        self.data = None
        self.n_samples = None
        self.n_features = None
        self.missing_mask = None
        # Cache: dictionary { observed_features_tuple: (mst_matrix, observed_data, valid_rows) }
        self.cache = {}
    
    def fit(self, X):
        """
        Store dataset and missing mask.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data with possible NaNs.
        """
        self.data = np.array(X, dtype=float)
        self.n_samples, self.n_features = self.data.shape
        self.missing_mask = np.isnan(self.data)
    
    def transform(self, X):
        """
        Impute missing values.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Same data or new data with missing values.

        Returns:
        --------
        X_imputed : np.array
            Imputed data, same shape as X.
        """
        # We assume transform is run on the same data shape as in fit
        X = np.array(X, dtype=float)
        X_imputed = X.copy()

        for i in range(self.n_samples):
            if self.missing_mask[i].any():
                X_imputed[i] = self._impute_row(i)
        return X_imputed
    
    def fit_transform(self, X):
        """
        Fit on X, then transform X in one call.
        """
        self.fit(X)
        return self.transform(X)
    
    def _impute_row(self, row_index):
        x = self.data[row_index]
        missing_features = self.missing_mask[row_index]    # boolean mask of which features are missing
        observed_features = ~missing_features              # boolean mask of which features are observed
    
        # If this row has no observed features, we cannot impute anything
        if not observed_features.any():
            return x  # remains all NaNs
    
        # Use a cache key based on which features are observed
        cache_key = tuple(observed_features)
        if cache_key in self.cache:
            mst_matrix, observed_data, valid_rows = self.cache[cache_key]
        else:
            # Build the MST for rows that have no missing values among the observed features
            observed_data = self.data[:, observed_features]
            # valid_rows picks out rows that are fully valid in *this* subspace of features
            valid_rows = ~np.isnan(observed_data).any(axis=1)
    
            if not valid_rows.any():
                return x  # no valid row to compare, return as is
    
            # Compute pairwise distances among valid rows in the subspace observed_features
            vdata = observed_data[valid_rows]
            diffs = vdata[:, np.newaxis, :] - vdata[np.newaxis, :, :]
            distances = np.sqrt((diffs ** 2).sum(axis=2))
    
            # Build MST
            mst_matrix = minimum_spanning_tree(distances)
    
            # Cache results to avoid rebuilding MST for the same missing pattern
            self.cache[cache_key] = (mst_matrix, observed_data, valid_rows)
    
        # Now find row_index's position in the MST's valid_rows
        valid_indices = np.where(valid_rows)[0]
        root_positions = np.where(valid_indices == row_index)[0]
        if len(root_positions) == 0:
            return x  # can't impute if row i isn't among valid_rows
    
        root_index_in_valid = root_positions[0]
    
        # BFS on MST to find up to n neighbors
        n_reachable_indices = self._get_n_reachable_points(
            mst_matrix, root_index_in_valid, self.n_neighbors
        )
        neighbor_rows = valid_indices[n_reachable_indices]
    
        # Use the neighbor rows to impute each missing feature
        x_imputed = x.copy()
        for feature_idx in np.where(missing_features)[0]:
            neighbor_values = self.data[neighbor_rows, feature_idx]
            neighbor_values = neighbor_values[~np.isnan(neighbor_values)]
            if neighbor_values.size > 0:
                x_imputed[feature_idx] = neighbor_values.mean()
    
        return x_imputed


    def _get_n_reachable_points(self, mst_matrix, root_index, n):
        """
        Traverse MST in BFS order (ignoring edge weights) from a given root node.

        Parameters:
        -----------
        mst_matrix : csr_matrix
            MST represented as a sparse matrix.
        root_index : int
            Index of the root node (in the valid_rows subspace).
        n : int
            Number of neighbor nodes to retrieve.

        Returns:
        --------
        reachable_indices : list of int
            Indices (in MST space) of up to n reachable nodes by BFS.
        """
        from collections import deque

        n_nodes = mst_matrix.shape[0]
        visited = [False] * n_nodes
        adjacency_list = [set() for _ in range(n_nodes)]

        # Convert MST to adjacency structure
        coo = mst_matrix.tocoo()
        for i, j in zip(coo.row, coo.col):
            adjacency_list[i].add(j)
            adjacency_list[j].add(i)  # undirected MST

        # BFS
        queue = deque([root_index])
        visited[root_index] = True
        reachable_indices = []

        while queue and len(reachable_indices) < n:
            current = queue.popleft()
            # Don’t include the root in the neighbor list itself
            if current != root_index:
                reachable_indices.append(current)
            for neighbor in adjacency_list[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return reachable_indices[:n]
