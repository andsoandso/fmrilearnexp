"""usage: see between -h"""
import sys, os
import argparse
import numpy as np

from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import permutation_test_score

from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.labels import create_y

from fmrilearnexp.common import extract_trial_features 
from fmrilearnexp.common import load_dimreduce_data_fromcl
from fmrilearnexp.common import extract_trial_features 

# =====
# Process the args, load and concatenate the data, do any preprocessing and 
# classify it, writing the results as name
# =====

# ----
# Setup
# ----
prng = np.random.RandomState(42)

parser = argparse.ArgumentParser(
        description=("Classify .csv data between experiments"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "name", 
        help="Name of this exp"
        )
parser.add_argument("-o", nargs=1,
        help="The name of the file to *append* the classfier results to"
        )
# Train args
parser.add_argument(
        "--train", nargs="+", 
        help="Targets, i.e. csv files to train on"
        )
parser.add_argument(
        "--train_data", default='0:5', type=str,
        help="Range of train data cols 'min:max' Uses python-style slicing."
        )
parser.add_argument(
        "--train_labels", default=6, type=int, 
        help="Column for train labels"
        )
parser.add_argument("--train_trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing"
        )
parser.add_argument("--train_window", default='0:15', type=str,
        help="Window to extract stats from (temporal centering)"
        )
# Test args
parser.add_argument("--test", nargs="+", 
        help="Targets, i.e. csv files to test on"
        )
parser.add_argument("--test_data", default='0:5', type=str,
        help="Range of data cols 'min:max' Uses python-style slicing."
        )
parser.add_argument("--test_labels", default=6, type=int, 
        help="Column for labels"
        )
parser.add_argument("--test_trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing"
        )
parser.add_argument("--test_window", default='0:15', type=str,
        help="Window to extract stats from (temporal centering)"
        )
# Other
parser.add_argument("--null", default=100, type=int, 
        help="N permutations")
parser.add_argument(
        "--clf", default="RandomForestClassifier", type=str,
        help="Select a classfier"
        )
args = parser.parse_args()


# Get data
def _data(csvs, feature_index, window, label_col, trial_tr_col):
    """Data loading and feature selection helper."""
    Xs, ys, indices, cvcodes = load_dimreduce_data_fromcl(
            csvs, feature_index, label_col, 
            label_col, trial_tr_col
            )

    X = np.concatenate(Xs)
    y = create_y(np.concatenate(ys)) 
    index = np.concatenate(indices)
    cvcode = np.concatenate(cvcodes)

    X, index, othermeta = extract_trial_features(X, index,  
            window, [y, cvcode], None
            )
    y, _ = othermeta  ## toss cvcode, not applicable
        
    return X, y


Xtrain, ytrain = _data(
        args.train,
        range(*[int(i) for i in args.train_data.split(':')]),
        range(*[int(i) for i in args.train_window.split(':')]),
        args.train_labels, 
        args.train_trial_tr
        )
Xtest, ytest = _data(
        args.test,
        range(*[int(i) for i in args.test_data.split(':')]),
        range(*[int(i) for i in args.test_window.split(':')]),
        args.test_labels, 
        args.test_trial_tr
        )
X = np.vstack([Xtrain, Xtest])
y = np.concatenate([ytrain, ytest])
cvcode = np.asarray([0]*ytrain.shape[0] + [1]*ytest.shape[0])
assert checkX(X)
assert X.shape[0] == y.shape[0], "X and y length mismatch"
assert X.shape[0] == cvcode.shape[0], "X and cvcode length mismatch"

# CV
cv = StratifiedKFold(cvcode, n_folds=2, indices=True)

# Classifier
if args.clf == "RandomForestClassifier":
    clf = RandomForestClassifier(
            n_estimators=500, max_features=None
            )
elif args.clf == "GradientBoostingClassifier":   
    clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, 
            max_depth=1, random_state=prng
            )
else:
    raise ValueError("--clf not understood")

# Go!
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

