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
y = iris.target

y_names = iris.target_names

test_ids = np.random.permutation(len(X))
#keeping the last 10 etries for testing, rest for training

X_train = X[test_ids[:-10]]
X_test = X[test_ids[-10:]]
y_train = y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

print(pred)
print(y_test)
print(accuracy_score(pred,y_test)*100)
