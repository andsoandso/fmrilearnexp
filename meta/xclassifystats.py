"""usage: see xclassifystats -h
 
    Note:
    ----
    All flags are optional and default to the values above.
"""
import sys, os
import argparse
from copy import deepcopy

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LeavePLabelOut
from sklearn.cross_validation import cross_val_score

from sklearn.feature_selection import RFECV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from fmrilearn.preprocess.data import checkX
from fmrilearn.cross_validation import KFoldChunks
from fmrilearnexp.common import _create_X_y_xcs

# =====
# Process the args, load and concatenate the data, do any preprocessing and 
# classify it, writing the results as name
# =====

# ----
# Setup
# ----
parser = argparse.ArgumentParser(
        description=("Cross-dataset classification"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Metadata setup
parser.add_argument("name", help="Name of this exp")

# Train args
parser.add_argument("--train_data", default='0:5', type=str,
        help="Range of train data cols 'min:max' Uses python-style slicing.")
parser.add_argument("--train_labels", default=6, type=int, 
        help="Column for trian labels")
parser.add_argument("--train_trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing")
parser.add_argument("--train", nargs="+", 
        help="Targets, i.e. csv files to train on")

# Test args
parser.add_argument("--test_data", default='0:5', type=str,
        help="Range of data cols 'min:max' Uses python-style slicing.")
parser.add_argument("--test_labels", default=6, type=int, 
        help="Column for labels")
parser.add_argument("--test_trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing")
parser.add_argument("--test", nargs="+", 
        help="Targets, i.e. csv files to test on")

# Output args
parser.add_argument("-o", nargs=1,
        help="The name of the file to *append* the classfier results to")

# Run NULL experiments?
parser.add_argument("--null", default=0, type=int, 
        help="Classify and save NULL distributions")

args = parser.parse_args()

# ----
# Get and process the data
Xtrain, ytrain = _create_X_y_xcs(args.train, args.train_data, 
        args.train_labels, args.train_trial_tr)
Xtest, ytest = _create_X_y_xcs(args.test, args.test_data, 
        args.test_labels, args.test_trial_tr)

# ----
# Classify, score, and save.
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
        max_depth=1, random_state=0)

clf.fit(Xtrain, ytrain)
ypredict = clf.predict(Xtest)

accs = accuracy_score(ytest, ypredict)
f = open(args.o[0], "a")
f.write("{0},{1},{2}\n".format(    
    np.round(np.mean(accs), decimals=3), 
    np.round(1.0/len(np.unique(ytrain)), decimals=3),
    args.name))
f.close()

# ----
# Do null runs?
if args.null < 0:
    raise ValueError("--null numt be 0 or greater")

null_accs = []
if args.null > 0:
    print("\tRunning NULL classifcations ({0} iterations)".
            format(args.null))

    ytrain_null = deepcopy(ytrain)
    for i in range(args.null):
        np.random.shuffle(ytrain_null)

        clf.fit(Xtrain, ytrain_null)
        ypredict_null = clf.predict(Xtest)

        null_accs.append(accuracy_score(ytest, ypredict_null))

f = open("null_"+args.o[0], "a")
f.write("{0},{1},{2},{3}\n".format(    
    np.round(np.mean(null_accs), decimals=3), 
    np.round(np.std(null_accs), decimals=3), 
    np.round(1.0/len(np.unique(ytrain_null)), decimals=3),
    args.name))
f.close()

