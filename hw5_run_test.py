#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:03:43 2020

hw5_run_test.py

This program runs the identified best classifier on the test dataset
Bagging w/ Decision Trees (31 estimators)

@author: HyunJae Pi, hyunpi@brandeis.edu
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing#from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# training data
df0 = pd.read_csv("./training2b.csv", header=None)
n_features = df0.shape[1]-1
X_training = preprocessing.scale(df0.loc[:, 0:n_features-1].values)
y_training = df0.loc[:, n_features].values

# test data
df1 = pd.read_csv("./test2b.csv", header=None)
X_test = preprocessing.scale(df1.loc[:, 0:n_features-1].values)

clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators = 31).fit(X_training, y_training)
y_test = clf.predict(X_test).astype(int)

# save
np.savetxt('./hw5_prediction.txt', y_test, fmt = '%d')
