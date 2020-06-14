# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:03:00 2020

@author: sarav
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

dataset = pd.read_csv('breast_cancer.csv')
x = dataset.iloc [:, 1:-1].values
y = dataset.iloc [:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



