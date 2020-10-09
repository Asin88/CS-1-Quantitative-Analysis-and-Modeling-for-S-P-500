# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 02:24:42 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
make_dataset.py
Script to load data and make dataset
"""

#Import modules
import pandas as pd 
import os

script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
script_dir1 = os.path.split(script_path)[0] #i.e. /path/to/dir/
script_dir2 = os.path.split(script_dir1)[0] #i.e. /path/to/dir/
cwd_dir = os.path.split(script_dir2)[0] #i.e. /path/to/
rel_path = "data\\raw\\cs-1.csv"
abs_file_path = os.path.join(cwd_dir, rel_path)
df_cs1 = pd.read_csv(abs_file_path, header='infer', index_col=None) 


#Formatting date values
df_cs1['date'] = pd.to_datetime(df_cs1.date) #,format='%Y-%m-%d')

#Sorting by date
df_cs1_sorted = df_cs1.sort_values(['date'],ascending=True,ignore_index=True)
#Creating new dataset to preserve raw dataset
df_cs1_new = df_cs1_sorted

#Check for missing values
df_missing = df_cs1_new[df_cs1_new.isna().any(axis=1)]

"""
There are only 11 rows with missing values. 8 rows have all data missing for
open, high and close. So they will be dropped. Out of remaining 3 rows, two
are missing Name tag, so they will be dropped. The remaining 1 row is also 
dropped as infering opening value from high, low, close is unreliable due to 
volatile nature of stock market data. 
"""
df_cs1_new.dropna(inplace = True)
df_cs1_new = df_cs1_new.reset_index()

#Separate date entities
#year = lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S" ).year
df_cs1_new['year'] = df_cs1_new['date'].dt.year #map(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S" ).year) 
df_cs1_new['month'] = df_cs1_new['date'].dt.month #map(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S" ).month)
df_cs1_new['week'] = df_cs1_new['date'].dt.isocalendar().week #map(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S" ).strftime('%V'))
df_cs1_new['dayOfWeek'] = df_cs1_new['date'].dt.dayofweek #map(lambda x: datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S" ).weekday())

#Identifier for whether a day is a Monday/Friday or Tuesday/Wednesday/Thursday
s_mon_fri = pd.Series([0]*df_cs1_new.shape[0])
for i in range(0,len(df_cs1_new)):
    if (df_cs1_new['dayOfWeek'][i] == 0 or df_cs1_new['dayOfWeek'][i] == 4):
        s_mon_fri.iat[i] = 1
    else:
        s_mon_fri.iat[i] = 0
df_cs1_new['mon_fri'] = s_mon_fri