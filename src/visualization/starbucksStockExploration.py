# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:39:13 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""

This script analyses the stock performance of a given stock from the data. 
The ticker nameo f the stock is given by user at runtime.

Objective 1: Weekly Volatility Index

    Output: b) Give an exploratory analysis on any one stock describing it’s 
    key statistical tendencies.
"""
"""
Import modules
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pandas.plotting import lag_plot
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

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


def plotFigures(df, title, xlabel, ylabel, fig_name):
    """
    This function builds, displays and saves line plots using given parameters.

    Arguments:
        df: dataframe to be plot
        title: title of plot
        xlabel: label of x-axis
        ylabel: lable of y-axis
        fig_name: name of figure to be saved

    Returns: None
    """
    df.plot()
    plt.title(f"{title}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    #    plt.savefig(f_getFilePath(f'reports\\figures\\{fig_name}.png'))
    pdf.savefig()
    plt.show()
    plt.close()


def plotHistograms(df, title, fig_name):
    """
    This function builds, displays and saves histogram using given parameters.

    Arguments:
        df: dataframe to be plot
        title: title of plot
        fig_name: name of figure to be saved

    Returns: None
    """
    df.plot.hist(figsize=(12, 12))
    plt.title(f"{title}")
    #    plt.savefig(f_getFilePath(f'reports\\figures\\{fig_name}.png'))
    pdf.savefig()
    plt.show()
    plt.close()


# Function to explore one stock
def f_exploreStock(df_cs1_new, ticker_name):

    """
    This function explores the market performance of a given stock.
    Objective 1 - Output b) Key Statistical Tendencies of given stock:
        Daily closing price and volume trend
        Distribution trend of opening, high, low, close and volume
        Daily returns
        Cumulative Returns
        Moving Average for 10,20,50 days
        Auto-correlation with 5 day lag
        Volatility
        Risk-Adjusted Returns
        Sharpe Ratio (or Risk-Adjusted Return Ratio)
        Sortino Ratio (or Downside Risk Adjusted Return Ratio)
    """

    print(f"Deep diving into performance of {ticker_name}...")  # print status

    df_tickername = df_cs1_new[df_cs1_new["Name"] == ticker_name]
    df_tickername = df_tickername.set_index("date")

    # Create PDF file for figures
    with PdfPages(f_getFilePath("reports\\figures\\Starbuck_Stock_Trend_Figures.pdf")) as pdf:

        # Daily closing price
        plotFigures(
            df_tickername[["close"]],
            f"{ticker_name} Closing Price Trend",
            "Time (in days)",
            "Closing Price",
            f"{ticker_name}_Daily_Closing_Price",
        )

        # Daily volume
        plotFigures(
            df_tickername[["volume"]],
            f"{ticker_name} Volume Trend",
            "Time (in days)",
            "Volume",
            f"{ticker_name}_Daily_Volume",
        )

        # Histograms
        titles = [
            f"{ticker_name} Opening Prices",
            f"{ticker_name} High Prices",
            f"{ticker_name} Low Prices",
            f"{ticker_name} Closing Prices",
            f"{ticker_name} Volume",
        ]
        fig_names = ["Open", "High", "Low", "Close", "Volume"]
        i = 0
        for column in df_tickername.columns:
            if column in ["open", "high", "low", "close", "volume"]:
                plotHistograms(df_tickername[column], f"{titles[i]}", f"{ticker_name}_{fig_names[i]}_Histogram")
                i = i + 1

        # Daily Returns
        df_tickername["Daily_Return"] = df_tickername["close"].pct_change()
        plotFigures(
            df_tickername["Daily_Return"],
            f"{ticker_name} Daily Return",
            "Time (in days)",
            "Daily Return",
            f"{ticker_name}_Daily_Return",
        )

        # Daily Returns - Histogram + KDE Plot
        sns.distplot(df_tickername["Daily_Return"].dropna(), bins=100, color="purple")
        plt.ylabel("Daily Return")
        plt.title(f"{ticker_name} Daily Return Histogram")
        #    plt.savefig(f_getFilePath(f'reports\\figures\\{ticker_name}_Daily_Return_HistKDE.png'))
        pdf.savefig()
        plt.show()
        plt.close()

        # Cumulative Return
        df_cr = df_tickername["Price_Returns"].cumsum()
        plotFigures(
            df_cr,
            f"{ticker_name} Cumulative Returns",
            "Time (in days)",
            "Cumulative Return",
            f"{ticker_name}_Cumulative_Return",
        )

        # Moving Average for 10, 20, 50 days
        ma_day = [10, 20, 50]

        for ma in ma_day:
            column_name = f"MA for {ma} days"
            df_tickername[column_name] = df_tickername["close"].rolling(ma).mean()

        plotFigures(
            df_tickername[["close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]],
            f"{ticker_name} Moving Average",
            "Time (in days)",
            "Closing Price",
            f"{ticker_name}_Moving_Average",
        )

        # Auto-Correlation
        plt.figure(figsize=(10, 10))
        lag_plot(df_tickername["open"], lag=5)
        plt.title(f"{ticker_name} Autocorrelation Plot")
        #    plt.savefig(f_getFilePath(f'reports\\figures\\{ticker_name}_Autocorrelation_Plot.png'))
        pdf.savefig()
        plt.show()
        plt.close()

        # Volatility
        plotFigures(
            df_tickername["Volatility"],
            f"{ticker_name} Volatility",
            "Time (in days)",
            "Historical Volatility",
            f"{ticker_name}_Volatility",
        )

        # Risk Analysis
        expected_return = df_tickername["Daily_Return"].mean()
        total_return = (
            df_tickername["Daily_Return"].iloc[-1] - df_tickername["Daily_Return"].iloc[1]
        ) / df_tickername["Daily_Return"].iloc[1]
        # annualized_return = (logsumexp((total_return + 1)**(1/6)))-1 #Using special scipy function because exponentail of very large negative number is rounded to zero.
        rf = 0.01  # risk-free return rate (assumed value)
        total_risk = df_tickername["Daily_Return"].std()
        # COnvert cumulative reurn to arithmetic returns
        a_return = df_cr.diff()
        # Plot returns vs risk
        x = df_tickername["Daily_Return"].mean()
        y = df_tickername["Daily_Return"].std()
        plt.scatter(x, y, s=np.pi * 20)
        plt.xlabel("Expected return")
        plt.ylabel("Risk")
        plt.annotate(
            f"{ticker_name}",
            xy=(x, y),
            xytext=(50, 50),
            textcoords="offset points",
            ha="right",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color="blue", connectionstyle="arc3,rad=-0.3"),
        )
        #    plt.savefig(f_getFilePath(f'reports\\figures\\{ticker_name}_Risk.png'))
        pdf.savefig()
        plt.show()
        plt.close()

    # Sharpe Ratio
    sharpe_ratio = ((a_return.mean() - rf) / a_return.std()) * np.sqrt(252)
    # Print report

    riskfile = open(f_getFilePath(f"reports\\Risk Analysis of {ticker_name}.txt"), "w+")
    print(f"Risk Analysis of {ticker_name}\n", file=riskfile)
    print("Expected Return: ", expected_return * 100, file=riskfile)
    print("Total Return: ", total_return * 100, file=riskfile)
    print("Total Risk: ", total_risk * 100, file=riskfile)
    print(f"Sharpe Ratio (assuminng {rf} risk-free return rate): ", sharpe_ratio, file=riskfile)
    if sharpe_ratio > 3:
        print("Stock performance is excellent. Extremely low volatility is estimated.", file=riskfile)
    elif sharpe_ratio > 2:
        print("Stock performance is very good. Very low volatility is estimated.", file=riskfile)
    elif sharpe_ratio > 1:
        print("Stock performance is good. Low volatility is estimated.", file=riskfile)
    elif sharpe_ratio > 0:
        print("Stock performance is fair. Moderate volatility is estimated.", file=riskfile)
    else:
        print("Stock performance is poor. High volatility is estimated.", file=riskfile)

    # Sortino Ratio
    # Create a downside return column with the negative returns only
    target = 0
    downside_returns = df_tickername.loc[df_tickername["Daily_Return"] < target]["Daily_Return"]
    # Calculate and std dev of downside
    down_risk = downside_returns.std()
    # Calculate the sortino ratio
    sortino_ratio = ((a_return.mean() - rf) / down_risk) * np.sqrt(252)

    # Print report
    print("\nDownside risk: ", down_risk * 100, file=riskfile)
    print("Sortino ratio: ", sortino_ratio, file=riskfile)
    if sortino_ratio > 3:
        print("Stock performance is excellent. Extremely low downside risk is estimated.", file=riskfile)
    elif sortino_ratio > 2:
        print("Stock performance is very good. Very low downside risk is estimated.", file=riskfile)
    elif sortino_ratio > 1:
        print("Stock performance is good. Low downside risk is estimated.", file=riskfile)
    elif sortino_ratio > 0:
        print("Stock performance is fair. Moderate downside risk is estimated.", file=riskfile)
    else:
        print("Stock performance is poor. High downside risk is estimated.", file=riskfile)

    riskfile.close()


"""
Run the script
"""
if __name__ == "__main__":

    print("Stock Analysis...")  # print status
