# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 03:05:45 2020

@author: AAKRITI
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 04:11:24 2020

@author: AAKRITI
"""

# Script to perform factor analysis

# Import modules
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# from sklearn.decomposition import PCA
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.dummy import DummyClassifier


# Define functions

# Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__)  # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0]  # i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0]  # i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


# Main funtion
print("Performing feature selection...")

from features.features import features, target

# Split data into train and test. Leave last 20% rows for test.
print("Splitting data into training and test datasets...")
split_ratio = 0.80
train_len = int(features.shape[0] * split_ratio)
train_x = features.iloc[:train_len, :]
test_x = features.iloc[train_len:, :]
train_y = target.iloc[:train_len]
test_y = target.iloc[train_len:]
print(train_x.columns)
print(train_y.head())

# Feature Selection
print("Performing Scaling and PCA...")

# Pipe flow is :
# Scaling the data -> PCA(Dimension reduction to two) --> Dummy Classifier
feature_pipe = Pipeline([("minmax", preprocessing.MinMaxScaler())], verbose=True)
# fitting the data in the pipe
feature_pipe.fit(train_x, train_y)
minmax = feature_pipe.named_steps["minmax"]
train_x_transformed = pd.DataFrame(minmax.transform(train_x))
# =============================================================================
# #PCA and Factor Analysis is encountering a bug related to incorrect calculation of variance.
# #Refer https://github.com/scikit-learn/scikit-learn/pull/9105
# num_factors = 10
# # Pipe flow is :
# # Scaling the data -> PCA(Dimension reduction to two) --> Dummy Classifier
# feature_pipe = Pipeline([('std', preprocessing.MinMaxScaler()), ('fa', FactorAnalyzer(n_factors = num_factors,rotation='varimax')), ('dummy_model', DummyClassifier("stratified",random_state = 42))], verbose = True)
# # fitting the data in the pipe
# feature_pipe.fit(train_x, train_y)
#
# #Transform training data
# fa = feature_pipe.named_steps['fa']
# eigen_values = pd.DataFrame(fa.get_eigenvalues()).round(decimals = 2)
# loadings = pd.DataFrame(fa.loadings_,).round(decimals = 2)
# communalities = pd.DataFrame(np.round(fa.get_communalities(), 2), columns = ['Communalities']).round(decimals = 2)
# covariances = pd.DataFrame(fa.get_factor_variance()).round(decimals = 2)
# train_x_transformed = pd.DataFrame(fa.transform(train_x))
#
# #Save factor analysis report
# fafile = open(f_getFilePath('reports\\factor_analysis.txt'), 'w+')
# print('Factor Analysis Report\n\n', file = fafile)
# print('Eigen Values', file = fafile)
# print(eigen_values, '\n', file = fafile)
# print('Factor Loadings', file = fafile)
# print(loadings, '\n', file = fafile)
# print(communalities, '\n', file = fafile)
# print("Variances \n1. Sum of squared loadings (variance) \n2. Proportional variance \n3. Cumulative variance)", file = fafile)
# print(covariances, file = fafile)
# fafile.close()
# =============================================================================
