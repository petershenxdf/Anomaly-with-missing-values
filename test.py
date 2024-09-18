#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 18:02:14 2024

@author: jiahao
"""
import cuml
from cuml import make_blobs
import cuml.cluster

# Generate synthetic data
X, y = make_blobs(n_samples=1000, n_features=2, centers=5, cluster_std=1.0)

# Perform k-means clustering with cuML (on GPU)
kmeans = cuml.cluster.KMeans(n_clusters=5)
kmeans.fit(X)

print("GPU used successfully.")
