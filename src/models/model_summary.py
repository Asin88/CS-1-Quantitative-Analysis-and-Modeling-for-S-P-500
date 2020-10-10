# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:01:03 2020

@author: AAKRITI
CS - 1 : Quantitative Analysis and Modeling for S&P 500

This script creates the summary of the trained model and calculates performance metrics.

Objective 3: Binary Classifier
"""

# Import modules
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics


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


# Function to print summary of final model
def f_modelSummary(final_model_name, final_model):
    """
    This functions prints the summary of the trained model.

    Arguments:
    final_model_name: trained model name
    final_model: trained model

    Returns: None
    """
    print("Final Model: ", final_model_name, "\n\n", file=sumfile)
    print(final_model.get_params(), file=sumfile)
    model1 = final_model.named_steps[final_model_name]
    if final_model_name == "Baseline":
        print("\n")

    elif final_model_name == "Logistic":
        print("\nIntercept:\n", model1.coef_)
        print("\nCoefficients:\n", model1.intercept_)

    elif final_model_name == "CART" or final_model_name == "RFC" or final_model_name == "GBC":
        # Get numerical feature importances
        importances = list(model1.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [
            (feature, round(importance, 2))
            for feature, importance in zip(list(train_x_transformed.columns), importances)
        ]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
        # Print out the feature and importances
        [print("Variable: {:20} Importance: {}".format(*pair), file=sumfile) for pair in feature_importances]
        feature_importances = pd.DataFrame(feature_importances, columns = ['Variables','Importance'])
    else:
        print("\n")


# Function to generate classification report
def f_classificationReport(train_y, train_pred):
    """
    This funtion gives the classification performace metrics: confusion matrix, precicion, recall, accuracy and f-score.

    Arguments:
        train_y: observed y
        train_pred: predicted y

    Returns: None
    """

    # Confusion Matrices
    confusionmatrix = metrics.multilabel_confusion_matrix(train_y, train_pred)

    # Find precision, recall, f score, support and accuracy
    class_report = metrics.classification_report(train_y, train_pred)
    accuracy = metrics.accuracy_score(train_y, train_pred)

    # Print to summary file
    print("Saving detailed classification report in summary file")
    print("\n\nClassification Report", file=sumfile)
    print("\nConfusion Matrices: \n", confusionmatrix, file=sumfile)
    print(class_report, file=sumfile)
    print("Accuracy: ", accuracy, file=sumfile)


# Funtion to plot ROC
def f_rocAUC(train_y, train_pred):
    """
    This function builds the ROC plot and calculates the ROC AUC metric.

    Arguments:
        train_y: observed y
        train_pred: predicted y

    Returns: None
    """

    #    Adding only unique labels of predited Y to remove warning of ill-defined precision and fscore.
    #    There are some labels in train_y, which dont appear in train_pred and hence it is ill-defined
    #    Selecting non-zero columns of train_pred
    m2 = (train_pred != 0).any()
    a = m2.index[m2]
    # Find AUC of ROC
    rocAUC = metrics.roc_auc_score(train_y, train_pred, labels=a, multi_class="ovr")
    print("\nROC AUC: ", rocAUC, file=sumfile)

    # Plot ROC curves for the multilabel problem

    # Compute ROC curve and ROC area for each class
    n_classes = train_y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(train_y.iloc[:, i], train_pred.iloc[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label="ROC curve of class {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multi-class")
    plt.legend(loc="lower right")
    plt.savefig(f_getFilePath("reports\\figures\\ROC_Curve.png"))
    print("\nROC Curve plot saved")
    plt.show()


from models.train_model import final_model_name, final_model, train_x_transformed, train_y, train_pred

sumfile = open(f_getFilePath("reports\\final_model_summary.txt"), "w+")
print("Writing model summary...")
f_modelSummary(final_model_name, final_model)
# Metrics
print("Calculating metrics...")
# if final_model_name == 'Logistic':
# Binarize labels
lb = preprocessing.LabelBinarizer()
train_y = lb.fit_transform(train_y)
train_y = pd.DataFrame(train_y, columns=["Low", "No_Confidence", "High"])
train_pred = pd.DataFrame(train_pred.astype(int))

f_classificationReport(train_y, train_pred)

f_rocAUC(train_y, train_pred)
sumfile.close()
