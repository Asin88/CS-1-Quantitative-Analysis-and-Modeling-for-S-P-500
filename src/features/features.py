# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 04:26:43 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""
This script builds the target and feature matrix for binary classification.

Objective 3: Binary Classification - Given a stock and it’s data, you have to
predict whether it will close lower than it opened (red) or higher than it
opened (green).
 
"""

"""
Import modules
"""
import os
import numpy as np
import pandas as pd

# Import functions from scripts
from report.write_report import f_addWorksheet

"""
Define functions
"""
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
    script_dir1 = os.path.split(script_dir)[0]  # i.e. /path/to/
    cwd_dir = os.path.split(script_dir1)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


"""
Objective 3: Binary Classification
    Target : closing price higher or lower than opening price. Variable needs
    to be created. 2 - higher, 1 - no change, 0 - lower
    Features: open, high, low, close, volume, price return, volatility, 
    year, month, week, dayofweek, mon-fri, Stock High minus Low price (H-L), 
    Stock Close minus Low price C-L), Stock High minus Close price (H-C),
    MA for 10, 20, 50 days, std dev for 7 days, today's close compared with 1 
    or 2 previous days'
"""

# Function to build features
def f_buildFeatures(df_ticker_pred):
    """This function builds features out of given dataset.

    Arguments:
        df_ticker_pred: dataset for which prediction is to be made
    Returns:
        features: Features dataframe
    Files: reports\\Output_Report.xlsx --> Worksheet: Features Snapshot
    """

    # Stock High minus Low price (H-L)
    df_ticker_pred["H-L"] = df_ticker_pred["high"] - df_ticker_pred["low"]
    # Stock Close minus Low price C-L)
    df_ticker_pred["C-L"] = df_ticker_pred["close"] - df_ticker_pred["low"]
    # Stock High minus Close price (H-C)
    df_ticker_pred["H-C"] = df_ticker_pred["high"] - df_ticker_pred["close"]
    df_ticker_pred["Daily_Return"] = df_ticker_pred["close"].pct_change()

    # Indicator for closing high or low for past two days
    df_cs1_lag1 = df_ticker_pred.shift(1)  # by 1 days
    df_cs1_lag2 = df_ticker_pred.shift(2)  # by 2 days
    df_ticker_pred["close_lag1"] = df_ticker_pred["close"] > df_cs1_lag1["close"]
    df_ticker_pred["close_lag2"] = df_ticker_pred["close"] > df_cs1_lag2["close"]

    # Moving Average for 10, 20, 50 days
    ma_day = [10, 20, 50]

    for ma in ma_day:
        column_name = f"MA for {ma} days"
        df_ticker_pred[column_name] = df_ticker_pred["close"].rolling(ma).mean()

    # std dev for 7 days
    df_ticker_pred["stdev for 7 days"] = df_ticker_pred["close"].rolling(7).std()

    df_ticker_pred = df_ticker_pred.replace([np.inf, -np.inf], np.nan)
    df_ticker_pred = df_ticker_pred.fillna(0)
    # Integer encoding for ticker name
    df_ticker_pred["Name"] = df_ticker_pred["Name"].astype("category")
    df_ticker_pred["Name_code"] = df_ticker_pred["Name"].cat.codes

    # Return features dataframe
    return df_ticker_pred

    # Features dataframe
    if "index" in df_ticker_pred.columns:
        df_ticker_pred = df_ticker_pred.drop("index", axis=1)
    features = df_ticker_pred.drop(["date", "Name"], axis=1)
    features.reset_index()

    # Return features dataframe
    return features


def f_targetFeatures(df_cs1_new, pairs, name_pred):
    """
    This function builds the target and feature matrices based on a selection of stocks strongly correlated with a given stock.

    Arguments:
        df_cs1_new: dataframe with input data
        pairs: dataframe wih 5 strongest pairs of each year in the dataset

    Returns:
        Target dataframe
        Features dataframe
    Files:
        worksheet with snapshot of target variable
        worksheet with description of target variable
        worksheet with snapshot of feature matrix
        worksheet with description of feature matrix
    """

    print("Building target and features...")  # print status

    # Select strong pairs of stocks containing given stock
    df_pairs = pairs[pairs["Pair1"] == name_pred]
    # Get stocks correlated strongly with given stocks
    df_corr_stocks = pd.DataFrame(df_pairs["Pair2"].unique())

    df_target = pd.DataFrame()
    df_features = pd.DataFrame()

    for corr_stock in df_corr_stocks:

        # Make dataframe
        df_ticker_all = df_cs1_new[df_cs1_new["Name"] == corr_stock]
        # using data till only 80% of days for training
        days_ratio = 0.80
        df_ticker_pred = df_ticker_all.iloc[: int((len(df_ticker_all) * days_ratio)), :]

        # Build target variable
        df_target = df_target.append(df_ticker_pred["close"] - df_ticker_pred["open"], ignore_index=True)

        # Build features
        df_features = df_features.append(f_buildFeatures(df_ticker_pred), ignore_index=True)

    df_target = df_target.mask(df_target > 0, 2)
    df_target = df_target.mask(df_target == 0, 1)
    df_target = df_target.mask(df_target < 0, 0)
    df_target = df_target.astype("category")
    df_target = df_target.rename(columns={0: "Observed"})

    # Return target and features dataframes
    return df_target, df_features


if __name__ == "__main__":
    print("Building features...")  # print status
