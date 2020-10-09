# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:44:55 2020

@author: AAKRITI
"""

#Import modules
import os
import datetime as dt
import pandas as pd

#Define functions

#Function to get absolute file path
def f_getFilePath(rel_path):
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
    cwd_dir = os.path.split(script_dir1)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return abs_file_path

#Enter ticker name
name_pred = 'SBUX'
#Enter date in format YYYY,MM,DD
date_pred = dt.datetime(2016,10,9)
#Enter location of file with historical price series for testing
df_hist_price = pd.read_csv(f_getFilePath('data\\test\\test.csv'), header='infer', index_col=None)

