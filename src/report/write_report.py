# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:39:04 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""

"""
This scripts adds a new output worksheet to /reports/Output_Report.xlsx
"""

"""
Import modules
"""
import os
import pandas as pd

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


# Function to add worksheet to output report
def f_addWorksheet(df, sheet_name):
    """
    This function adds a worksheet with data from given dataframe to /reports/Output_Report.xlsx.

    Arguments:
        df: dataframe contaiing reporting data
        sheet_name: name of new worksheet to be added

    Returns:
        None

    Files:
        /reports/Output_Reports.xlsx
    """
    with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), engine="openpyxl", mode="a") as writer:
        df.to_excel(writer, sheet_name=sheet_name)


"""
Run the script
"""
if __name__ == "__main__":

    print("Writing Report...")  # print status
