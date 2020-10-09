# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 04:26:43 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

Objective 3: Binary Classification - Given a stock and it’s data, you have to
predict whether it will close lower than it opened (red) or higher than it
opened (green).

This script builds the target and feature matrix for binary classification. 
"""

# Import modules
import os
import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

# import graphlab as gl
# import talib as ta

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
    script_dir1 = os.path.split(script_dir)[0]  # i.e. /path/to/
    cwd_dir = os.path.split(script_dir1)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


"""
Objective 3: Binary Classification
    Target : closing price higher or lower than opening price. Variable needs
    to be created. 1 - higher, 0 - lower
    Features: open, high, low, close, volume, price return, volatility, 
    year, month, week, dayofweek, mon-fri, Stock High minus Low price (H-L), 
    Stock Close minus Low price C-L), Stock High minus Close price (H-C),
    MA for 10, 20, 50 days, std dev for 7 days, today's close compared with 1 
    or 2 previous days', Moving Average Convergence Divergence (MACD), 
    Relative Strength Index
"""

# Import dataset from script
from visualization.calcVolatility import df_cs1_new

# select any stock from training dataset
name_pred = "ABT"

# Make dataframe
df_ticker_all = df_cs1_new[df_cs1_new["Name"] == name_pred]
#using data till only 80% of days for training
days_ratio = 0.80
df_ticker_pred = df_ticker_all.iloc[:int((len(df_ticker_all)*days_ratio)),:]

# Build features

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

# Features matrix
features = df_ticker_pred.drop(["index", "date", "Name"], axis=1)
with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), engine="openpyxl", mode="a") as writer:
    features.tail(30).to_excel(writer, sheet_name="Features Snapshot")

# Build target variable
target = df_ticker_pred["close"] - df_ticker_pred["open"]
target = target.mask(target > 0, 2)
target = target.mask(target == 0, 1)
target = target.mask(target < 0, 0)
target = target.astype("category")
with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), engine="openpyxl", mode="a") as writer:
    target.tail(30).to_excel(writer, sheet_name="Target Snapshot")
