# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 03:27:01 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script tests the classifier with new data. - Given a stock and it’s data, predict whether it will
close lower than it opened (red) or higher than it opened (green)

Input Arguments - Ticker Symbol, date (to predict for), Historical Price Series for the
selected stock (up till the mentioned date, but be sure to avoid look
ahead bias)

Function Returns - 1 (for Green), 0 (for Red), 0.5 (For No Confidence)

(A ‘No Confidence’ will be treated as a random prediction and is
better than a wrong prediction)
"""

#Import modules
import os
import datetime
import pandas as pd
import numpy as np

# Define functions
# Function to get file path
def f_getFilePath(rel_path):
    """
    This function gives the absolute file path from a given relative path

    Arguments:
        rel_path: relative path of file to be accessed

    Returns:
        absolute path of file to be accessed
    """
    script_path = os.path.abspath(__file__)  # i.e. /path/to/dir/foobar.py
    script_dir = os.path.split(script_path)[0]  # i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path

"""
Test Data is entered here.
"""
#Name of stock for which prediction is to be made
name_pred = 'GOOG'
#Date in YYYY,MM,DD format for which prediction is to be made
date_pred = datetime.datetime(2016,10,9)
#HIstorical price series of given stock
test_data = pd.read_csv(f_getFilePath("data\\test\\test_data.csv"), header="infer", index_col=None)

#Format date column
test_data["date"] = pd.to_datetime(test_data.date)  

"""
Do not look ahead. Remove data beyond given date.
"""
New_data = test_data.loc[test_data['date']<date_pred]

"""
Drop missing data
"""
New_data.dropna(inplace=True)
New_data = New_data.reset_index()
"""
Transform new data with scaler
"""

"""
Build features
"""
#Compute the logarithmic returns using the Closing price
New_data["Price_Returns"] = np.log(New_data["close"] / New_data["close"].shift(1))
New_data.fillna(0, inplace=True)  # First row will be NA. Changing it to 0.

# Compute Volatility using the pandas rolling standard deviation function
New_data["Volatility"] = New_data["Price_Returns"].rolling(window=5).std() * np.sqrt(5)
New_data.fillna(0, inplace=True)  # First 4 rows will be NA. Changing to 0.
from data.make_dataset import f_dataFeatures
New_data_d = f_dataFeatures(New_data)
from features.features import f_buildFeatures
features = f_buildFeatures(New_data_d)
#Minmax scaler
from features.feature_selection import feature_pipe
feature_pipe.fit(features)
minmax = feature_pipe.named_steps["minmax"]
features_transformed = pd.DataFrame(minmax.transform(features))
#features_transformed = pd.DataFrame(minmax.transform(features))

"""
Get trained and tested model
"""
#Trained model
from models.train_model import final_model

"""
Use trained and tested model to predict with test data
"""
print(features_transformed.columns)
test_pred = final_model.predict_proba(features_transformed)  # Predicted probabilities of y
test_pred = pd.DataFrame(test_pred)
test_pred = test_pred.mask(test_pred <= 0.40, 0)
test_pred = test_pred.mask((test_pred > 0.40) & (test_pred <= 0.60), 0.5)
test_pred = test_pred.mask(test_pred > 0.60, 1)

"""
Give Prediction
"""
#Print prediction
Prediction = test_pred.iloc[(New_data.shape[0] + 1),:]
if Prediction[0] == 1:
    print('Prediction: 0 \nRed \nStock will close lower than open')
elif Prediction[1] == 1:
    print('Prediction: 0.5 \nNo Confidence \nStock may or may not close lower than open')
else:
    print(f'Prediction: 1 \nGreen \nStock will close higher than open')

# Model Summary and Merics
import models.model_summary

