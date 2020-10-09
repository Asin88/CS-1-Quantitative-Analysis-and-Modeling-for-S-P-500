# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:28:35 2020

@author: AAKRITI
CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script uses the trained model to make predictions on test split data. 

Objective 1: Binary CLassifier

    Predict whether closing will be higher or lower than opening price for a given stock on a given day.
"""

#Import modules
import os
import pandas as pd

#Define functions
#Function to get file path
def f_getFilePath(rel_path):
    """
    This function gives the absolute file path from a given relative path
    
    Arguments:
        rel_path: relative path of file to be accessed
    
    Returns:
        absolute path of file to be accessed
    """
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
    script_dir1 = os.path.split(script_dir)[0] #i.e. /path/to/
    cwd_dir = os.path.split(script_dir1)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)

from features.feature_selection import test_x, test_y, minmax, final_model

"""
The features in test dataset are first scaled with MinMax to bring in the range of 0 to 1. 
Then dimensionality is reduced using Factor Analysis.

***NOTE: Due to bug issues with dimension reduction techniques (PCA and Factor Analysis), this step has not been performed for the time being.***
"""
#Transform test_x with MinMax scaling and Factor Analysis  
test_x_transformed = minmax.transform(test_x)

#Predict 
test_pred = final_model.predict_proba(test_x_transformed) #Predicted probabilities of y
test_pred = pd.Series(test_pred)
test_pred = pd.cut(test_pred, 3, labels=[0,0.5,1], include_lowest=True)

#Model Summary and Merics
import models.model_summary