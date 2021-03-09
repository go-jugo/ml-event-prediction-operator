from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn import datasets
import sklearn 
import statistics
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from monitoring.time_it import timing
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xlsxwriter
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.svm import LinearSVC, SVC
from openpyxl import load_workbook
from types import SimpleNamespace
import ast
import glob
from math import ceil


@timing
def eval(X, clf):
    result = clf.predict(X)
    return result




