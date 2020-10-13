# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 03:27:01 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""

This script tests the classifier with new data. - Given a stock and it’s data, predict whether it will
close lower than it opened (red) or higher than it opened (green)

Input Arguments - Ticker Symbol, date (to predict for), Historical Price Series for the
selected stock (up till the mentioned date, but be sure to avoid look
ahead bias)

Function Returns - 1 (for Green), 0 (for Red), 0.5 (For No Confidence)

(A ‘No Confidence’ will be treated as a random prediction and is
better than a wrong prediction)
"""

"""
Import modules
"""
import os
import datetime
import pandas as pd
import numpy as np

# Import functions from scripts
from visualization.calcVolatility import f_weeklyVolatilityIndex
from features.features import f_buildFeatures
from models.predict_model import f_testModel
from models.model_summary import f_classificationReport

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


# Function to predict with new data
def f_predictWithNewData(final_model, pairs):
    """
    This function uses the final model to make predictions on new data.

    Arguments:
        final model: Patatemers of final model

    Returns:
        None
    """
    """
    Test Data is entered here.
    """
    # Name of stock for which prediction is to be made
    name_pred = "GOOG"
    # Date in YYYY,MM,DD format for which prediction is to be made
    date_pred = datetime.datetime(2016, 10, 9)
    # HIstorical price series of given stock
    test_data = pd.read_csv(f_getFilePath("data\\test\\test_data.csv"), header="infer", index_col=None)

    # Format date column
    test_data["date"] = pd.to_datetime(test_data.date)

    """
    Drop missing data
    """
    New_data.dropna(inplace=True)
    New_data = New_data.reset_index()

    """
    Build features
    """
    # Add weekly volatility index
    New_data = f_weeklyVolatilityIndex(New_data)

    # Select strong pairs of stocks containing given stock
    df_pairs = pairs[pairs["Pair1"] == name_pred]
    # Get stocks correlated strongly with given stocks
    df_corr_stocks = pd.DataFrame(df_pairs["Pair2"].unique())

    for corr_stock in df_corr_stocks:
        # Make dataframe
        df_ticker_all = New_data[New_data["Name"] == corr_stock]
        # Build features
        New_data = New_data.append(f_buildFeatures(New_data), ignore_index=True)
    """
    Do not look ahead. Remove data beyond given date.
    """
    New_data = test_data.loc[test_data["date"] < date_pred]
    New_data_test = test_data.loc[test_data["date"] <= date_pred]

    """
    Make prediction
    """
    New_test_pred = f_testModel(New_data, New_data_test, final_model, feature_pipe)

    # Rename class names according to desired output
    New_test_pred = New_test_pred.mask(New_test_pred == 1, 0.5)
    New_test_pred = New_test_pred.mask(New_test_pred == 2, 1)
    """
    Print Prediction for given date
    """
    # Print prediction
    Prediction = New_test_pred[New_test_pred["date"] == date_pred]
    if Prediction == 0:
        print("Prediction: 0 \nRed \nStock will close lower than open")
    elif Prediction == 0.5:
        print("Prediction: 0.5 \nNo Confidence \nStock may or may not close lower than open")
    else:
        print(f"Prediction: 1 \nGreen \nStock will close higher than open")

    # Prediction Summary and Metrics
    lb = preprocessing.LabelBinarizer()
    New_data_test = lb.fit_transform(New_data_test)
    New_data_test = pd.DataFrame(New_data_test, columns=["Low", "No_Confidence", "High"])
    f_classificationReport(New_data_test, New_test_pred)


if __name__ == "__main__":
    print("Predicting with new data...")  # print status
