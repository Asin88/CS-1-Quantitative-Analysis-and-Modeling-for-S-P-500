# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:19:26 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
Main Program

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

# =============================================================================
# Import Modules
# =============================================================================

from datetime import datetime
import os
import pandas as pd

# =============================================================================
# #Define Functions
# =============================================================================

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


# #Main function
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
        reports/outfile.txt
        reports/Top_Ten_Most_and_Least_Volatile_Stocks.csv
        reports/Pair_Trading_Strongest_Pairs.csv

    Figures saved at reports/figures/

    """

    # Initialize output file
    # Report title
    d_report = {\
            "Title": "Project: CS - 1 : Quantitative Analysis and Modeling for S&P 500",
             "Author": "Aakriti Sinha",
             "Created on": '06-10-2020'
             "Last run on": datetime.now()
             }
    df_report = pd.DataFrame(d_report)
    with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), mode="w+") as writer:
        df_report.to_excel(writer, sheet_name="Title Page")

    # ------------------------------------------------------------------------------
    # Raw Data
    # Get raw dataframe
    from data.make_dataset import df_cs1_new

    # Describe raw data
    with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), engine="openpyxl", mode="a") as writer:
        df_cs1_new.head(30).to_excel(writer, sheet_name="Data Snapshot")
        df_cs1_new.describe().to_excel(writer, sheet_name="Data Description")

    # ------------------------------------------------------------------------------
    """
    Objective 1: A Weekly Volatility Index
    """
    # Calculate Volatility and Rank Stocks
    import visualization.calcVolatility  # import df_cs1_new

    # Describe Starbucks Stock
    import visualization.starbucksStockExploration

    """
    Objective 2: Pairs Trading
    """
    # Pair Trading
    import visualization.pairsTrading

    """
    Objective 3: Binary Classification
    """
    # Build features
    import features.features
    import features.feature_selection

    # Train model
    import models.train_model

    # Make predictions
    import models.predict_model


# Main program
if __name__ == "__main__":

    print("Main function")
    # Output file
    outfile = open(f_getFilePath("reports\\outfile.txt"), "w+")
    # Call main function
    main()
    # Close output file
    outfile.close()
