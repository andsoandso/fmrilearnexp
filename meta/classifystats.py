"""usage: see classifystats -h

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
from fmrilearn.preprocess.labels import create_y
from fmrilearn.preprocess.data import create_X_stats

# =====
# Process the args, load and concatenate the data, do any preprocessing and 
# classify it, writing the results as name
# =====

# ----
# Setup
# ----
parser = argparse.ArgumentParser(
        description="Run RFE using Linear SVMs on the supplied csv data")
parser.add_argument("name", help="Name of this exp")
parser.add_argument("--data", default='0:5', type=str,
        help="Range of data cols 'min:max' Uses python-style slicing.")
parser.add_argument("--labels", default=6, type=int, 
        help="Column for labels")
parser.add_argument("--index", default=7, type=int, 
        help="Column number for cross_validation codes")
parser.add_argument("--trial_tr", default=8, type=int,
        help="Column number for trial level TR indexing")
parser.add_argument("--window", default='0:15', type=str,
        help="Window to extract stats from (temporal centering)")
parser.add_argument("-t", nargs="+", 
        help="Targets, i.e. csv files to concastnate then classify")
parser.add_argument("-o", nargs=1,
        help="The name of the file to *append* the classfier results to")
parser.add_argument("--null", default=0, type=int, 
        help="Classify and save NULL distributions")
parser.add_argument("--clf", default="GradientBoostingClassifier", type=str,
        help="Select a classfier")
args = parser.parse_args()
feature_index = range(*[int(i) for i in args.data.split(':')])
window =  range(*[int(i) for i in args.window.split(':')])

# ----
# Load
# ----
# Check data exists
csvs = args.t
for csvn in csvs:
    if not os.path.exists(csvn):
        raise IOError("{0} does not exist".format(csvn))

X, y, index, tTR = None, None, None, None
for csvn in csvs:
    dftmp = pd.read_csv(csvn, sep=",", header=0)

    ## Reinit 
    if (X is None) and (y is None):
        X = np.asarray(dftmp.ix[:,feature_index])
        y = np.asarray(dftmp.ix[:,args.labels], dtype=np.str)
        index = np.asarray(dftmp.ix[:,args.index])
        tTR = np.asarray(dftmp.ix[:,args.trial_tr])
    ## Otherwise stack
    else:
        X = np.vstack([X, np.asarray(dftmp.ix[:,feature_index])])
        y = np.concatenate([y, np.asarray(dftmp.ix[:,args.labels], 
                dtype=np.str)])
        index = np.concatenate([index, np.asarray(dftmp.ix[:,args.index])])
        tTR = np.concatenate([tTR, np.asarray(dftmp.ix[:,args.trial_tr])])

# ----
# Preproces
# ----
# Convert y to integer codes
y = create_y(y)

# Convert trial_tr to a trial_index fot stat gen
cnt = 0
last_ttr = 0
trial_index = []
for ttr in tTR:
    if ttr < last_ttr:  
        ## If the TR is not monotonic, new trial
        ## is detected so inc cnt
        cnt += 1
    trial_index.append(cnt)
    last_ttr = deepcopy(ttr)

# Sane so far?
assert checkX(X)
assert X.shape[0] == y.shape[0], "X and y length mismatch"

# ----
# Convert X to Xstat
# ----

# Init
trials = np.unique(trial_index)
Xmax = np.zeros((trials.shape[0], X.shape[1]))
Xmin = np.zeros_like(Xmax)
#Xmean = np.zeros_like(Xmax)
#Xvar = np.zeros_like(Xmax)

ystat = []
indexstat = []
for ii, trial in enumerate(trials):
    # Locate this trials data
    # Get time to peak/min
    # And their diff
    # Finally get trial means and variances
    mask = trial == trial_index
    x_trial = X[mask,:]

    Xmax[ii,:] = np.argmax(x_trial[window,], axis=0)
    Xmin[ii,:] = np.argmin(x_trial[window,], axis=0)
    Xdiff = Xmax - Xmin
    #Xmean[ii,:] = x_trial[window,].mean(axis=0)
    #Xvar[ii,:] = x_trial[window,].var(axis=0)

    # Only need one label for each trial
    ystat.append(y[mask][0])
    indexstat.append(index[mask][0])

# And rename
X = np.hstack([Xmax, Xmin, Xdiff])
#X = np.hstack([Xmax, Xmin, Xdiff, Xmean, Xvar])
y = np.asarray(ystat)
index = np.asarray(indexstat)
del Xmax, Xmin, Xdiff#, Xmean, Xvar ## Cleanup ASAP

# Still sane?
assert checkX(X)
assert X.shape[0] == y.shape[0], "X and y length mismatch"

# Use trial chunks of labels to CV?
if args.labels == args.index:
    cv = KFold(y.shape[0], n_folds=5, indices=True)
else:
    # ----
    # To split by subject
    unindex = sorted(np.unique(index))
    chunks = np.repeat(-1, index.shape[0])   
        ## Ensures no data are missed during recode.
    for k, ui in enumerate(unindex):
        chunks[ui == index] += np.int(k + 1)   
            ## +1 to adj for -1 init
    cv = LeavePLabelOut(chunks, p=1)

    # ----
    # Sane?
    assert X.shape[0] == chunks.shape[0], "X and chunks length mismatch"       
    assert np.sum(chunks == -1) == 0, "Chunks is malformed"  

# ----
# Classify
# ----

# ----
# Using GradientBoostingClassifier
if args.clf == "RandomForestClassifier":
    clf = RandomForestClassifier(n_estimators=500, max_features=None)
elif args.clf == "GradientBoostingClassifier":   
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
            max_depth=1, random_state=0)
else:
    raise ValueError("--clf not understood")

accs = cross_val_score(clf, X, y=y,
        scoring="accuracy", cv=cv,
        n_jobs=1, verbose=0,
        fit_params=None, score_func=None)

f = open(args.o[0], "a")
f.write("{0},{1},{2}\n".format(    
    np.round(np.mean(accs), decimals=3), 
    np.round(1.0/len(np.unique(y)), decimals=3),
    args.name))
f.close()
# ----

# ----
# Do null runs?
if args.null < 0:
    raise ValueError("--null must be 0 or greater")

null_accs = []
if args.null > 0:
    print("\tRunning NULL classifcations ({0} iterations)".
            format(args.null))
    ynull = deepcopy(y)
    for i in range(args.null):
        np.random.shuffle(ynull)
        accs = cross_val_score(clf, X, y=ynull, scoring="accuracy", cv=cv,
                n_jobs=1, verbose=0,
                fit_params=None, score_func=None)

        null_accs.append(np.mean(accs))

    f = open("null_"+args.o[0], "a")
    f.write("{0},{1},{2},{3}\n".format(    
        np.round(np.mean(null_accs), decimals=3), 
        np.round(np.std(null_accs), decimals=3), 
        np.round(1.0/len(np.unique(ynull)), decimals=3),
        args.name))
    f.close()


