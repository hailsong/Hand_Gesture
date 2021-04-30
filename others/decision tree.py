from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from IPython.display import Image

import pandas as pd
import numpy as np
import pydotplus

import os

tennis_data = pd.read_csv('C:/myPyCode/data/playtennis.csv')
print(tennis_data)