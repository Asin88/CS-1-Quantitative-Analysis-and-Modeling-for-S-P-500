# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:17:53 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script calculates the weekly volatility of stocks and identifies the top
ten most and least volatile stocks per week.

Objective 1: Establish a weekly volatility index which ranks stocks on the basis of intraday price movements.
"""

"""
Import modules
"""
import pandas as pd
import numpy as np
import os

# Import functions from scripts
from report.write_report import f_addWorksheet

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


def f_weeklyVolatilityIndex(df_cs1_new):

    """
    This function calculates teh weekly volatitilty index.
    Objective 1: Weekly Volatility Index

        Volatility can be measured by the standard deviation of returns
        over a chosen period of time. Historical Volatility (HV) is calculted as:

            HV = (Std dev of Price Returns) * (Square root (T))

        where T is the number of periods in chosen time frame. Price Returns can
        be calculated as the natural logarithm of price change over a single period.

        For weekly VIX, price returns are calculated using daily price change.
        The number of periods in a week is 5 (counting only weekdays as the stock
        market is closed on weekends).

    Arguments:
        df_cs1_new: dataframe with processed data

    Returns:
        modified dataframe with new columns for price returns and weekly volatility index
    """

    print("Calculating Weekly Volatility Index...")  # print status

    # Compute the logarithmic returns using the Closing price
    df_cs1_new["Price_Returns"] = np.log(df_cs1_new["close"] / df_cs1_new["close"].shift(1))
    # First row will be NA. Changing it to 0.
    df_cs1_new.fillna(0, inplace=True)

    # Compute Volatility using the pandas rolling standard deviation function
    df_cs1_new["Volatility"] = df_cs1_new["Price_Returns"].rolling(window=5).std() * np.sqrt(5)
    # First 4 rows will be NA. Changing to 0.
    df_cs1_new.fillna(0, inplace=True)
    df_cs1_new = df_cs1_new.sort_values(["date", "Volatility"], ascending=True, ignore_index=True)

    # Return modified dataframe
    return df_cs1_new


def f_rankByVolatility(df_cs1_new):
    """
    This functions finds the top 10 most and least volatile stocks of each week.
    Objective 1 - Output c): The output needs to be grouped weekly showing the Top 10 Most and Least Volatile stocks.
    Ranking is done on the basis of the mean value of weekly volatility on last day of week - Friday for each stock.

    Arguments:
        df_cs1_new: dataframe with volatility index data
    Returns:
        None
    Files:
        worksheet with ranking output added to /reports/Output_Report.xlsx
    """

    print("Ranking stocks by Weekly Volatility Index...")  # print status

    # Group by week with frequency of every Friday and then by name
    Name_group = df_cs1_new.groupby([pd.Grouper(key="date", freq="W-FRI"), "Name"]).agg({"Volatility": "mean"})
    g = Name_group["Volatility"].groupby(level=0, group_keys=False)
    # Find top 10 most volatile stocks
    most_vol_10 = pd.DataFrame(g.nlargest(10)).reset_index()
    # Find top 10 least volatile stocks
    least_vol_10 = pd.DataFrame(g.nsmallest(14)).iloc[4:].reset_index()

    # Print report
    output_1a = pd.DataFrame()
    output_1a["Year"] = most_vol_10["date"].dt.year
    output_1a["Week"] = most_vol_10["date"].dt.isocalendar().week
    output_1a["Most Volatile Stocks"] = most_vol_10["Name"]
    output_1a["Volatility Top Ten"] = most_vol_10["Volatility"]
    output_1a["Least Volatile Stocks"] = least_vol_10["Name"]
    output_1a["Volatility Bottom Ten"] = least_vol_10["Volatility"]
    # Print report
    f_addWorksheet(output_1a, "Top_10_Volatile_Stocks")


"""
Run the script
"""
if __name__ == "__main__":

    print("Volatility Index...")  # print status
