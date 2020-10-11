# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:19:26 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
Main Program

Github Repository: https://github.com/Asin88/CS-1-Quantitative-Analysis-and-Modeling-for-S-P-500

Objective 1: establish a weekly volatility index which ranks stocks on the
basis of intraday price movements.

     to be calculated on a weekly time frame and both intraday as well as
     weekly change in price needs to be used in calculating volatility)

    The index should rank the stocks from most to least volatile in the
    selected time frame.

    Output: a) The output needs to be grouped weekly showing the Top 10 Most
    and Least Volatile stocks.

    b) Give an exploratory analysis on any one stock describing it’s
    key statistical tendencies.

Objective 2: Pairs Trading
    Identify the 5 strongest pairs for every year in the dataset (eg. 5
    strongest pairs for 2014, 2015 and so on)

Objective 3: Binary Classification - Given a stock and it’s data, you have to
predict whether it will close lower than it opened (red) or higher than it
opened (green).

"""

"""
Import Modules
"""
from datetime import datetime
import os
import pandas as pd
#Import functions from scripts
from data.make_dataset import f_loadmodeldata
from report.write_report import f_addWorksheet
from visualization.calcVolatility import f_weeklyVolatilityIndex
from visualization.calcVolatility import f_rankByVolatility
from visualization.starbucksStockExploration import f_exploreStock
from visualization.pairsTrading import f_pairsTrading    
from features.features import f_targetFeatures
from features.feature_selection import f_splitData

"""
Define Functions
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
    cwd_dir = os.path.split(script_dir)[0]  # i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path


#Main function
def main():
    """
    This is the main function.It calls scripts to perform analysis.

    Scripts called:
        data/make_dataset.py
        visualization/calcVolatility.py
        visualization/starbucksStockExploration.py
        visualization/pairsTrading.py
        models/train.py
        models/predict.py
        visualization/visualize.py

    Reports generated:
        reports\\Output_Report.xlsx

    Figures saved at reports/figures/

    """

    # Initialize output file
    """
    Make a new excel file at /reports/ to store outputs in separate worksheets. 
    """
    #Title page
    d_report = {"Title": "Project: CS - 1 : Quantitative Analysis and Modeling for S&P 500",
             "Author": "Aakriti Sinha",
             "Created on": '06-10-2020',
             "Last run on": datetime.now()}
    df_report = pd.DataFrame.from_dict(d_report, orient = 'index')
    with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), mode="w+") as writer:
        df_report.to_excel(writer, sheet_name="Title Page")
   
    """
    Load raw data and process it.
    """
    # Get processed dataframe
    df_cs1_new = f_loadmodeldata('data\\raw\\cs-1.csv')
    
    # Describe data
    """
    Report basic statistics of the data.
    """
    f_addWorksheet(df_cs1_new.head(30), 'Data_Snapshot')
    f_addWorksheet(df_cs1_new.describe(), 'Data_Description')
            
    """
    Objective 1: A Weekly Volatility Index
    """
    # Calculate Volatility 
    f_weeklyVolatilityIndex(df_cs1_new)
    # Rank Stocks
    f_rankByVolatility(df_cs1_new)
    
    """
    Objective 1b: Explore any one stock's performance
    """
    # Select stock data
    # Get ticker name from user
     print('Enter ticket name to be analysed: ')
     loopagain = 1
     while loopagain == 1:
        ticker_name = input()
        if ticker_name not in df_cs1_new['Name'].unique():
            print('Stock not listed in dataset. Please try another stock.')
        else:
            loopagain = 0
    
    # Describe selected stock
    f_exploreStock(df_cs1_new,ticker_name)
    
    """
    Objective 2: Pairs Trading
    """
    # Pair Trading
    output_2 = f_pairsTrading(df_cs1_new)

    """
    Objective 3: Binary Classification
    """
    
    # Build features
    #Select any stock to train data on 
    name_pred = 'ABT'
    target, features = f_targetFeatures(df_cs1_new, output_2, name_pred)
    #Print target decription
    f_addWorksheet(target.tail(30), "Target_Snapshot")
    f_addWorksheet(target.describe(), "Target_Description")
    #Print features decription
    f_addWorksheet(features.tail(30), "Features_Snapshot")
    f_addWorksheet(features.describe(), "Features_Description)
    
    #Select features
    train_x, test_x, train_y, test_y = f_splitData(features, target)
    from features.feature_selection import f_featureSelection
    train_x_transformed = f_featureSelection(train_x, train_y)

    # Train model
    import models.train_model

    # Make predictions
    import models.predict_model


# Main program
if __name__ == "__main__":

    print("Main function")
    main()
    