# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 05:18:51 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script calculates the weekly volatility of stocks and identifies the top
ten most and least volatile stocks per week.

Objective 2: Pairs Trading
    Identify the 5 strongest pairs for every year in the dataset (eg. 5
    strongest pairs for 2014, 2015 and so on)
"""

# Import modules
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


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


from visualization.calcVolatility import df_cs1_new

output_2 = pd.DataFrame()

# Get CLosing value for each Stock for each year
for year in df_cs1_new["year"].unique():
    # Select year
    df = df_cs1_new[df_cs1_new["year"] == year]
    # Select only name and colsing price columns
    new_df = pd.DataFrame(df["Name"])
    new_df["close"] = df["close"]
    # Get list of [Name,closing price of each day of selected year]
    l = [[label] + grp["close"].unique().tolist() for label, grp in new_df.groupby("Name")]
    df1 = pd.DataFrame(l)
    # Make dataframe of names as columns and daily closing price as rows
    stock_closing = df1.T
    stock_closing.columns = stock_closing.iloc[0]
    stock_closing = stock_closing[1:]
    stock_closing = stock_closing.astype(float)

    # Build correlation matrix for each year
    corr = stock_closing.corr()
    # Select only upper triangle of matrix
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Sort by correlation coefficient
    corr = corr.unstack().transpose().sort_values(ascending=False, kind="quicksort").dropna()
    # Format correlation dataframe
    corr = corr.rename_axis(["Pair1", "Pair2"])
    corr = corr.reset_index()
    corr = corr.rename(columns={0: "Corr. Coeff."})
    corr.index = [year] * corr.shape[0]
    corr = corr.rename_axis(["Year"])
    corr = corr.reset_index()
    # Append to output dataframe
    output_2 = output_2.append(corr[0:5])


# Print report
# print(corr.head())
output_2.to_csv(f_getFilePath("reports\\Pair_Trading_Strongest_Pairs.csv"), index=False, encoding="utf-8")
