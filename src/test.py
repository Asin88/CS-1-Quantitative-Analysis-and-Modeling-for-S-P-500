# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 02:38:52 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
Main Program

This script is the test script. It uses new unknown data to predict whether the 
given stock will cloe higher or lower than opening price. 

Input Arguments - Ticker Symbol, date (to predict for), Historical Price Series for the
selected stock (up till the mentioned date, but be sure to avoid look
ahead bias)

Function Returns - 1 (for Green), 0 (for Red), 0.5 (For No Confidence)

(A ‘No Confidence’ will be treated as a random prediction and is
better than a wrong prediction)
"""

from data.test_data import name_pred, date_pred, df_hist_price

