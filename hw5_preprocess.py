#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 11:26:06 2020

hw5_preprocess.py

This program converts letters to numeric values according to EIIP representation.
 ’G’ to 0.0806, ’A’ to 0.1260, ’C’ to 0.1340, ’T’ to 0.1335

@author: HyunJae Pi, hyunpi@brandeis.edu
"""

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder

# convert string to array
def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# convert letter to number (EIIP representation -- see hw5 report)
def letter_to_number(my_array):
    tmp = label_encoder.transform(my_array)
    num = tmp.astype(float)
    num[num == 0] = .1260 # A
    num[num == 1] = .1340 # C
    num[num == 2] = .0806 # G
    num[num == 3] = .1335 # T
    num[num == 4] = .00 # anything else
    return num

# label encoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))

# 1. convert training data
df0 = pd.read_csv("./training.data", header=None)
df0.columns=["class", "seq_letter"]

n_rows0 = df0.shape[0]
n_cols0 = df0.shape[1]
#print(n_rows0, n_cols0)

#test_sequence = 'AACGCGCTTNN'
#print(letter_to_number(string_to_array(test_sequence)))



tmp_data0 = np.zeros((n_rows0, 60))
for i in range(0, n_rows0):
    tmp_data0[i, :] = letter_to_number(string_to_array(df0.loc[i,'seq_letter'])).T
print(tmp_data0.shape)

tmp_class = np.array(df0["class"])
training_data = np.column_stack((tmp_data0, tmp_class))

# save
df_save = pd.DataFrame(training_data)
df_save.to_csv('training2b.csv', index=False, header=None)


# 2. convert test data
df1 = pd.read_csv("./test.data", header=None)
df1.columns=["seq_letter"]

n_rows1 = df1.shape[0]
n_cols1 = df1.shape[1]
tmp_data1 = np.zeros((n_rows1, 60))
for i in range(0, n_rows1):
    tmp_data1[i, :] = letter_to_number(string_to_array(df1.loc[i,'seq_letter']))
print(tmp_data1.shape)

# save
df_save = pd.DataFrame(tmp_data1)
df_save.to_csv('test2b.csv', index=False, header=None)
