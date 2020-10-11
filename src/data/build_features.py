# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 19:16:13 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500
"""
"""
This script adds new features to the original processes data.
"""
"""
Import modules
"""
import os
import pandas as pd
#Import modules from script
from data.make_dataset import f_processData
from visualization.calcVolatility import f_weeklyVolatilityIndex

"""
Define functions
"""
 def f_addFeatures(df):
     f_processData(df)
     f_weeklyVolatilityIndex(df)
    
"""
Run the script
"""      
if __name__ == "__main__":

    print('Adding new features...') #print status
    
    
