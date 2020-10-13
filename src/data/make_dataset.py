# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:24:42 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""
This script loads data and processes raw data to create a clean dataframe.
"""
"""
Import modules
"""
import pandas as pd
import os

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
    script_dir1 = os.path.split(script_path)[0]  # i.e. /path/to/dir/
    script_dir2 = os.path.split(script_dir1)[0]  # i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir2)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


# Function to load data into dataframe
def f_loadData(file_loc):
    """
    This functions loads data from csv file located in given folder.

    Arguments:
        file_loc: relative path of file to be loaded
    Returns:
        df_data: dataframe with raw data
    """
    # print status
    print("Loading data from csv file...")
    # Load data from csv file
    df_data = pd.read_csv(f_getFilePath(file_loc), header="infer", index_col=None)
    # Return dataframe with raw data
    return df_data


# Function to process data
def f_processData(df_cs1):
    """
    This function performs basic processing on the data to format datatypes, handle missing values and build date features
    """
    print("Processing Data...")  # print status

    # Formatting date values
    df_cs1["date"] = pd.to_datetime(df_cs1.date)

    # Sorting by date
    df_cs1_sorted = df_cs1.sort_values(["date"], ascending=True, ignore_index=True)

    # Creating new dataset to preserve raw dataset
    df_cs1_new = df_cs1_sorted

    # Check for missing values
    print("Checking for missing values...")  # print status
    df_missing = df_cs1_new[df_cs1_new.isna().any(axis=1)]
    # The current dataset is quite clean and missing data is rare. For small number of missing values, imputation is not required. Moreover, in case of missing Name, imputation is not helpful either. Thus, rows with missing data are being dropped.
    if not df_missing.empty:
        print("Dropping rows with missing values...")  # print status
        df_cs1_new.dropna(inplace=True)
        df_cs1_new = df_cs1_new.reset_index()

    # Create date features
    df_cs1_new = f_dataFeatures(df_cs1_new)

    # Return processed dataframe
    return df_cs1_new


# Function to separate date entities
def f_dataFeatures(df_cs1_new):
    """
    This function creates new date features as dummy variables.

    Arguments:
        df_cs1_new: dataframe

    Returns:
        df_cs1_new: modified dataframe with dummy date features
    """
    # Make new column with year
    df_cs1_new["year"] = df_cs1_new["date"].dt.year
    # Make new column with month
    df_cs1_new["month"] = df_cs1_new["date"].dt.month
    # Make new column with week
    df_cs1_new["week"] = df_cs1_new["date"].dt.isocalendar().week
    # Make new column with day of week
    df_cs1_new["dayOfWeek"] = df_cs1_new["date"].dt.dayofweek

    """
    Whether a day is a Monday/Friday or a mid-week day could be an influence on stock performance.
    """
    # Identifier for whether a day is a Monday/Friday or Tuesday/Wednesday/Thursday
    s_mon_fri = pd.Series([0] * df_cs1_new.shape[0])
    for i in range(0, len(df_cs1_new)):
        if df_cs1_new["dayOfWeek"][i] == 0 or df_cs1_new["dayOfWeek"][i] == 4:
            s_mon_fri.iat[i] = 1
        else:
            s_mon_fri.iat[i] = 0
    df_cs1_new["mon_fri"] = s_mon_fri

    # Return modified dataframe
    return df_cs1_new


# Function to load and process model data
def f_loadmodeldata(file_loc):
    """
    This function loads and processes the data for model building

    Arguments:
        file_loc: relative path of csv file containing raw data

    Returns:
        dataframe with processed data
    """
    # Load data from csv located at /data/raw/
    df_cs1 = f_loadData(file_loc)

    # Process data
    df_cs1_new = f_processData(df_cs1)
    """
    There are only 11 rows with missing values. 8 rows have all data missing for open, high and close. So they will be dropped. Out of remaining 3 rows, two are missing Name tag, so they will be dropped. The remaining 1 row is also dropped as infering opening value from high, low, close is unreliable due to olatile nature of stock market data.
    """
    # Return processed dataframe
    return df_cs1_new


"""
Run the script
"""
if __name__ == "__main__":
    # Print status
    print("Making dataset...")
