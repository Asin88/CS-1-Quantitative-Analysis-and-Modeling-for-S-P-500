# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 03:05:45 2020

@author: AAKRITI
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 04:11:24 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""
This script performs feature selection.
"""
"""
Import modules
"""
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from factor_analyzer import FactorAnalyzer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

"""
Define functions
"""

# Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__)  # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0]  # i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0]  # i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


def f_splitData(features, target):
    """
    This function splits the data into train and test data. Since we have time series data, the split will be done so that the last 20% days are held out for testing."

    Arguments:
        features: dataframe with feaures
        target: dataframce with target
    Returns:
        train_x: training features
        train_y: training target
        test_x: testing features
        test_y: testing target
    """
    print("Splitting data into training and test datasets...")
    split_ratio = 0.80
    train_len = int(features.shape[0] * split_ratio)
    train_x = features.iloc[:train_len, :]
    test_x = features.iloc[train_len:, :]
    train_y = target.iloc[:train_len]
    test_y = target.iloc[train_len:]

    print(train_x.head())
    print(train_y.head())

    # Return train and test data
    return train_x, test_x, train_y, test_y


# Function for Feature Selection
def f_featureSelection(train_x, train_y):

    """This function performs scaling and Factor analysis on training data.

    Arguments:
        train_x: training features
        train_y: training target

    Return:
        train_x_transformed: scaled and factor analysed features
    """
    print("Performing Scaling and Feature Selection...")

    #    # Pipe flow is :
    #    # Scaling the data
    #    print("Skipping feature selection. Only scaling data...") #print status
    #    feature_pipe = Pipeline([("minmax", preprocessing.MinMaxScaler())], verbose=True)
    num_factors = 10
    # Pipe flow is :
    # Scaling the data -> Factor Analysis (Dimension reduction to ten)
    print("Scaling and Factor Analysis...")  # print status
    feature_pipe = Pipeline(
        [("std", preprocessing.MinMaxScaler()), ("fa", FactorAnalyzer(n_factors=num_factors, rotation="varimax"))],
        verbose=True,
    )
    # Pipe flow is :
    # Scaling the data -> RFE using Decision trees
    #    print("Scaling and RFE using decision trees...") #print status
    #    feature_pipe = Pipeline([('std', preprocessing.MinMaxScaler()), ('rfe', RFE(estimator=DecisionTreeClassifier(), n_features_to_select=num_factors))], verbose = True)
    #
    # fitting the data in the pipe
    feature_pipe.fit(train_x, train_y)

    # Apply scaler transform
    minmax = feature_pipe.named_steps["minmax"]
    train_x_scaled = pd.DataFrame(minmax.transform(train_x))

    """
    PCA and Factor Analysis is encountering a bug related to incorrect calculation of variance.
    Refer https://github.com/scikit-learn/scikit-learn/pull/9105
    """

    # Transform training data
    fa = feature_pipe.named_steps["fa"]
    eigen_values = pd.DataFrame(fa.get_eigenvalues()).round(decimals=2)
    loadings = pd.DataFrame(
        fa.loadings_,
    ).round(decimals=2)
    communalities = pd.DataFrame(np.round(fa.get_communalities(), 2), columns=["Communalities"]).round(decimals=2)
    covariances = pd.DataFrame(fa.get_factor_variance()).round(decimals=2)
    train_x_transformed = pd.DataFrame(fa.transform(train_x_scaled))

    # Save factor analysis report
    fafile = open(f_getFilePath("reports\\factor_analysis.txt"), "w+")
    print("Factor Analysis Report\n\n", file=fafile)
    print("Eigen Values", file=fafile)
    print(eigen_values, "\n", file=fafile)
    print("Factor Loadings", file=fafile)
    print(loadings, "\n", file=fafile)
    print(communalities, "\n", file=fafile)
    print(
        "Variances \n1. Sum of squared loadings (variance) \n2. Proportional variance \n3. Cumulative variance)",
        file=fafile,
    )
    print(covariances, file=fafile)
    fafile.close()

    # Return transformed features
    return train_x_transformed, feature_pipe


"""
Run the script
"""
if __name__ == "__main__":
    print("Performing feature selection...")  # print status
