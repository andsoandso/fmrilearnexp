"""usage: see within -h"""
import sys, os
import argparse
from copy import deepcopy

import pandas as pd
import numpy as np
            
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import permutation_test_score

from wheelerexp.common import extract_trial_features 
from wheelerexp.common import load_dimreduce_data_fromcl

from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.labels import create_y

# ----
# argv process
parser = argparse.ArgumentParser(
        description="Classify .csv data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "name", 
        help="Name of this exp"
        )
parser.add_argument(
        "-t", nargs="+", 
        help="Targets, i.e. csv files to concatenate then classify"
        )
parser.add_argument(
        "-o", nargs=1,
        help="The name of the file to *append* the classfier results to"
        )
parser.add_argument(
        "--data", default='0:5', type=str,
        help="Range of data cols 'min:max' Uses python-style slicing."
        )
parser.add_argument(
        "--labels", default=6, type=int, 
        help="Column for labels"
        )
parser.add_argument(
        "--cv", default=7, type=int, 
        help="Column number for cross_validation codes"
        )
parser.add_argument(
        "--trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing"
        )
parser.add_argument(
        "--window", default='0:15', type=str,
        help="Window to extract stats from (temporal centering)"
        )
parser.add_argument(
        "--null", default=0, type=int, 
        help="N permutations"
        )
parser.add_argument(
        "--clf", default="RandomForestClassifier", type=str,
        help="Select a classfier"
        )
args = parser.parse_args()
prng = np.random.RandomState(42)


# ----
# Load and preprocess data
feature_index = range(*[int(i) for i in args.data.split(':')])

csvs = args.t
# load_dimreduce_data_fromcl(csvs, feature_cols, label_col, cv_col, trial_tr_col)
Xs, ys, indices, cvcodes = load_dimreduce_data_fromcl(
        csvs, feature_index, args.labels, 
        args.cv, args.trial_tr
        )
X = np.concatenate(Xs)
y = create_y(np.concatenate(ys)) 
index = np.concatenate(indices)
cvcode = np.concatenate(cvcodes)

window =  range(*[int(i) for i in args.window.split(':')])
X, index, othermeta = extract_trial_features(
        X, index,  window, [y, cvcode], None
        )
y, cvcode = othermeta

assert X.shape[0] == y.shape[0], "X and y length mismatch"

# ----
# Setup CV: 
#
# Kfold splitting by label or by custom label
# (probably by participant)
n_fold = 3
if args.labels == args.cv:
    cv = KFold(y.shape[0], n_folds=n_fold, indices=True)
else:
    uncvcode = np.unique(cvcode)
    uncvcode.sort()
    chunks = np.repeat(-1, cvcode.shape[0])   
    for k, ui in enumerate(uncvcode):
        chunks[ui == cvcode] += np.int(k + 1)   ## +1 to adj for -1 init
        
    # cv = LeavePLabelOut(chunks, p=len(uncvcode)/n_fold) ## int div
    cv = StratifiedKFold(chunks, n_folds=n_fold, indices=True)

    assert X.shape[0] == chunks.shape[0], "X and chunks length mismatch"       
    assert np.sum(chunks == -1) == 0, "Chunks is malformed"  

# Clf
if args.clf == "RandomForestClassifier":
    clf = RandomForestClassifier(n_estimators=500, max_features=None)
elif args.clf == "GradientBoostingClassifier":   
    clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, 
            max_depth=1, random_state=prng
            )
else:
    raise ValueError("--clf not understood")

# Go
acc, perm, p = permutation_test_score(
        clf, X, y, 
        score_func=None, 
        cv=cv, 
        n_permutations=args.null, 
        n_jobs=5, 
        labels=None, 
        random_state=prng, 
        verbose=0, 
        scoring="accuracy"
        )

# Save
f = open(args.o[0], "a")
f.write("{0},{1},{2},{3}\n".format(    
        np.round(acc, decimals=3), 
        np.round(np.mean(perm), decimals=3),
        np.round(p, decimals=4),
        args.name)
        )
f.close()

