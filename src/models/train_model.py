# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:11:32 2020

@author: AAKRITI

CS - 1 : Quantitative Analysis and Modeling for S&P 500

Objective 3: Binary Classification - Given a stock and it’s data, you have to
predict whether it will close lower than it opened (red) or higher than it
opened (green).

This script trains and selects the best classification model.
Models compared:
    Baseline
    Multinomial Logistic Regression
    K Nearest Neighbours
    Gaussian Naive Bayes
    Decision Tree Classifier (CART)
    Random Forest Classifier
    Gradient Boosting Classifier
"""

#Import modules
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline 
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

#Define functions
#Function to get file path
def f_getFilePath(rel_path):
    """
    This function gives the absolute file path from a given relative path
    
    Arguments:
        rel_path: relative path of file to be accessed
    
    Returns:
        absolute path of file to be accessed
    """
    script_path = os.path.abspath(__file__) # i.e. /path/to/dir/foobar.py
    script_dir = os.path.split(script_path)[0] #i.e. /path/to/dir/
    script_dir1 = os.path.split(script_dir)[0] #i.e. /path/to/
    cwd_dir = os.path.split(script_dir1)[0] #i.e. /path/to/
    abs_file_path = os.path.join(cwd_dir, rel_path)
    return(abs_file_path)
    
def f_buildModels(m_name, clf_estimator, train_x, train_y):
    """
    This function builds several classifer models and fits training data.
    
    Arguments:
        m_name: model name
        clf_estimator: classification estimator
        train_x: training features
        train_y: training target
        
    Returns: None
    """
    print(f'Building {m_name} model...')
    model_pipe = Pipeline([(m_name, clf_estimator)], verbose = True) 
    # fitting the data in the pipe 
    clf_model = model_pipe.fit(train_x, train_y)
    return [m_name, clf_model]
    
#Function to evaluate all training models
def f_modelEvaluation(model_name, clf_model, train_x, train_y):
    """
    This function cross-validates the model passed as argument with 5-fold Time Series Split and measures average accuracy of predictions in each fold.
    
    Arguments:
    model_name: model name
    clf_model: classificaion model fitted on training data
    train_x: triaing features
    train_y: training target
    
    Returns: two lists
        list holding model name and mean accuracy
        list holding model name and mean macro f1 score        
    """
    #Define the evaluation method
    cv = TimeSeriesSplit(n_splits=5)
    # evaluate the model on the dataset
    n_scores_acc = cross_val_score(clf_model, train_x, train_y, scoring='accuracy', cv=cv, n_jobs=-1)
    n_scores_f1 = cross_val_score(clf_model, train_x, train_y, scoring='f1_macro', cv=cv, n_jobs=-1)
    return [model_name,clf_model, np.mean(n_scores_acc)], [model_name,clf_model, np.mean(n_scores_f1)]
    
#Main function
print('Training binary classifier models...')

from features.feature_selection import train_x_transformed,train_y

#Build training models
print('Building training models...')
modelevalfile = open(f_getFilePath('reports\\model_comparison.txt'),'w')
print('Model Comparison\n\n',file = modelevalfile)
print('Model Name\tAccuracy\tROC AUC', file = modelevalfile)

estimator_dict = \
              {\
               'Baseline': DummyClassifier(strategy = "stratified",random_state = 42),
#               'Logistic': LogisticRegression(multi_class = 'multinomial'),
               'KNN3': KNeighborsClassifier(n_neighbors=3),
               'KNN5': KNeighborsClassifier(n_neighbors=5),
               'KNN7': KNeighborsClassifier(n_neighbors=7),
               'GNB': GaussianNB(),
               'CART':DecisionTreeClassifier(criterion="entropy",max_depth = 3),
               'RFC': RandomForestClassifier(n_estimators=10, max_depth=3),
               'GBC': GradientBoostingClassifier()
              }
print(train_y.head(30))        
model_dict = {}
for m_name, clf_estimator in estimator_dict.items():
    m_name, clf_model = f_buildModels(m_name, clf_estimator, train_x_transformed, train_y)
    model_dict[m_name] = clf_model
    
#Evaluate and Compare All Models
print('Comparing all models...')
model_accs = []
model_f1s = []
for m_name, clf_model in model_dict.items():
    list_acc, list_f1 = f_modelEvaluation(m_name, clf_model, train_x_transformed, train_y)
    model_accs.append(list_acc)
    model_f1s.append(list_f1)
    
#Report performance
print(model_accs)
print(model_f1s)
df_temp= pd.DataFrame(model_accs, columns = ['Model Name','Model Params', 'Accuracy'])
df_temp = df_temp.drop('Model Params', axis = 1) 
df_temp2 = pd.DataFrame(model_f1s, columns = ['Model Name','Model Params', 'F1-Score'])
df_temp2 = df_temp2.drop('Model Params', axis = 1)  
df_temp['F1-Score'] = df_temp2['F1-Score']
with pd.ExcelWriter(f_getFilePath("reports\\Output_Report.xlsx"), engine="openpyxl", mode = 'a') as writer:
    df_temp.to_excel(writer, sheet_name='Model Comparison')

#Select model with highest macro f1 scores
"""Investigation of model performance metrics (precision and recall) showed that predictions are heavily biased towards 0.
    This indicates class imbalance in training data. Accuracy, precision and recall are sensitive to class imbalance. 
    However, f1 score is insenstive. Hence, the evaluation score is selected as f1 score.
"""
model_f1s.sort(key = lambda x: x[2], reverse = True)
final_model_name = model_f1s[0][0]
final_model = model_f1s[0][1]
print(f'Selected model is {final_model_name}.\nF1 score is {model_f1s[0][2]}')

#Predict from train_x_transformed
train_pred = final_model.predict_proba(train_x_transformed) #Predicted probabilities of y
#if final_model_name == 'Logistic':
classes = [0,1,2]
train_pred_1 = pd.Series(train_pred.astype(int).dot(classes))
#else: 
#    train_pred = pd.Series(train_pred)
#    train_pred = pd.cut(train_pred, 3, labels=[0,1,2], include_lowest=True)
    
#Model Summary and Metrics
import models.model_summary #import f_modelSummary(final_model_name, final_model)
