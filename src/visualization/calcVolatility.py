# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:17:53 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script calculates the weekly volatility of stocks and identifies the top
ten most and least volatile stocks per week.

Objective 1: Establish a weekly volatility index which ranks stocks on the 
basis of intraday price movements.
"""

# Import modules
import pandas as pd
import numpy as np
import os

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
Objective 1: Weekly Volatility Index

    Volatility can be measured by the standard deviation of returns 
    over a chosen period of time. Historical Volatility (HV) is calculted as:
        
        HV = (Std dev of Price Returns) * (Square root (T))
        
    where T is the number of periods in chosen time frame. Price Returns can 
    be calculated as the natural logarithm of price change over a single period.
    
    For weekly VIX, price returns are calculated using daily price change. 
    The number of periods in a week is 5 (counting only weekdays as the stock
    market is closed on weekends). 
"""
# Import data from script
from data.make_dataset import df_cs1_new

# Compute the logarithmic returns using the Closing price
df_cs1_new["Price_Returns"] = np.log(df_cs1_new["close"] / df_cs1_new["close"].shift(1))
df_cs1_new.fillna(0, inplace=True)  # First row will be NA. Changing it to 0.

# Compute Volatility using the pandas rolling standard deviation function
df_cs1_new["Volatility"] = df_cs1_new["Price_Returns"].rolling(window=5).std() * np.sqrt(5)
df_cs1_new.fillna(0, inplace=True)  # First 4 rows will be NA. Changing to 0.
df_cs1_new = df_cs1_new.sort_values(["date", "Volatility"], ascending=True, ignore_index=True)

"""
Objective 1 - Output c): The output needs to be grouped weekly showing the 
Top 10 Most     and Least Volatile stocks.
Ranking on mean value of Volatility on last day of week - Friday for each stock
"""
Name_group = df_cs1_new.groupby([pd.Grouper(key="date", freq="W-FRI"), "Name"]).agg({"Volatility": "mean"})
g = Name_group["Volatility"].groupby(level=0, group_keys=False)
most_vol_10 = pd.DataFrame(g.nlargest(10)).reset_index()
#    most_vol_10.rename(columns={"Name": "Most Volatile Stocks", 'Volatility':'Volatility Top Ten'}, inplace = True)
least_vol_10 = pd.DataFrame(g.nsmallest(14)).iloc[4:].reset_index()
#    least_vol_10.rename(columns={"Name": "Least Volatile Stocks", 'Volatility':'Volatility Bottom Ten'}, inplace = True)

output_1a = pd.DataFrame()
output_1a["Year"] = most_vol_10["date"].dt.year
output_1a["Week"] = most_vol_10["date"].dt.isocalendar().week
output_1a["Most Volatile Stocks"] = most_vol_10["Name"]
output_1a["Volatility Top Ten"] = most_vol_10["Volatility"]
output_1a["Least Volatile Stocks"] = least_vol_10["Name"]
output_1a["Volatility Bottom Ten"] = least_vol_10["Volatility"]
# Print report
output_1a.to_csv(f_getFilePath("reports\\Top_Ten_Most_and_Least_Volatile_Stocks.csv"), index=False, encoding="utf-8")
