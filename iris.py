# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 01:50:55 2019

@author: mr_ro
"""

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

X = iris.data