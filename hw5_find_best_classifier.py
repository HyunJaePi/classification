#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:07:29 2020

hw5_find_best_classifier.py

Using preprocessed training data, this program 
tests different classifiers (SVM, Decision Trees, AdaBoost, and Bagging) 
and identifies the best one.

@author: HyunJae Pi, hyunpi@brandeis.edu
"""

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier

# select_params in 1D or 2D parameter space
def select_params(X, y, clf, param1_name, param1, param2_name ="", param2 =[], nfolds = 5):
    if param2_name == "":
        param_grid = {param1_name: param1}
    else:
        param_grid = {param1_name: param1, param2_name : param2}
    grid_search = GridSearchCV(clf, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# 0. load a preprocessed training data
df0 = pd.read_csv("./training2b.csv", header=None)
n_features = df0.shape[1]-1
X = preprocessing.scale(df0.loc[:, 0:n_features-1].values)
y = df0.loc[:, n_features].values

# 1. SVM w/ rbf
clf = svm.SVC(kernel = 'rbf')
param1 = [0.001, 0.01, 0.1, 1, 10] # C
param2= [0.001, 0.01, 0.1, 1] # gamma
best_params = select_params(X, y, clf, 'C', param1, 'gamma', param2)
print("1. SVM with rbf")
clf_svm = svm.SVC(kernel = 'rbf', C = best_params['C'], gamma = best_params['gamma']) #0.905
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_svm, X, y, cv = 5)))) # {'C': 1, 'gamma': 0.01}
print(best_params)

# 2. SVM w/ polynomial
clf = svm.SVC(kernel = 'poly')
param1 = [0.0001, 0.001, 0.01, 0.1, 1] # C
param2= [0.01, 0.1, 1, 10] # gamma
best_params = select_params(X, y, clf, 'C', param1, 'gamma', param2)
print("2. SVM with polynomial")
clf_svm = svm.SVC(kernel = 'poly', C = best_params['C'], gamma = best_params['gamma'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_svm, X, y, cv = 5)))) # 0.886
print(best_params) # {'C': 0.0001, 'gamma': 1}

# 3. SVM w/ linear
clf = svm.SVC(kernel = 'linear')
param1 = [ 0.1, 1, 10, 100] # C
param2= [] # gamma
best_params = select_params(X, y, clf, 'C', param1)
print("3. SVM with linear")
clf_svm = svm.SVC(kernel = 'linear', C = best_params['C'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_svm, X, y, cv = 5)))) #0.897
print(best_params) # {'C': 10}


# 4. Decision Tree
clf = DecisionTreeClassifier(random_state=0,  criterion='gini',splitter='best')
max_depths = np.linspace(1, 10, 10, endpoint=True) 
min_samples_splits = np.linspace(0.01, 0.2, 10, endpoint=True)
best_params = select_params(X, y, clf, 'max_depth', max_depths, 'min_samples_split', min_samples_splits)
print("4. Decision Trees\n")
clf_dt = DecisionTreeClassifier(random_state=0,  criterion='gini',splitter='best', max_depth = best_params['max_depth'], min_samples_split = best_params['min_samples_split'] )
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_dt, X, y, cv = 5)))) #0.924
print(best_params) #{'max_depth': 8.0, 'min_samples_split': 0.01}

##### ensemble classifiers
# base estimators
svc = svm.SVC(probability=True, kernel='linear')
dt = DecisionTreeClassifier()
n_estimators = np.linspace(30, 40, 10, endpoint = True, dtype = int)
learning_rate = np.linspace(0.1, 1, 10, endpoint = True)

# 5. AdaBoost w/ decision trees
clf =  AdaBoostClassifier(base_estimator=dt)
best_params = select_params(X, y, clf, 'n_estimators', n_estimators, 'learning_rate', learning_rate)
print("5. AdaBoost w/ Decision Trees")
clf_ab_t = AdaBoostClassifier(base_estimator=dt, n_estimators = best_params['n_estimators'], learning_rate = best_params['learning_rate'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_ab_t, X, y, cv = 5)))) # 0.927
print(best_params) # 0.6 & 35

# 6. AdaBoost w/ svm  -- too slow later
clf =  AdaBoostClassifier(base_estimator=svc)
best_params = select_params(X, y, clf, 'n_estimators', n_estimators, 'learning_rate', learning_rate)
print("6. AdaBoost w/ SVM")
clf_ab_s = AdaBoostClassifier(base_estimator=svc, n_estimators = best_params['n_estimators'], learning_rate = best_params['learning_rate'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_ab_s, X, y, cv = 5)))) #0.931
print(best_params) #0.9 40


svc = svm.SVC(kernel='linear')

# 7. Bagging w/ decision trees
clf = BaggingClassifier(base_estimator=dt, random_state=1)
best_params = select_params(X, y, clf, 'n_estimators', n_estimators)
print("7. Bagging w/ Decision Trees")
clf_bag_t = BaggingClassifier(base_estimator=dt, n_estimators = best_params['n_estimators'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_bag_t, X, y, cv = 5)))) #0.942
print(best_params) # 31

# 8. Bagging w/ svm
clf = BaggingClassifier(base_estimator=svc, random_state=1)
best_params = select_params(X, y, clf, 'n_estimators', n_estimators)
print("8. Bagging w/ SVM")
clf_bag_t = BaggingClassifier(base_estimator=svc, n_estimators = best_params['n_estimators'])
print("accuracy: %.3f" %(np.mean(cross_val_score(clf_bag_t, X, y, cv = 5)))) #0.898
print(best_params) # 33
