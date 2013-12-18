"""usage: classfiy name --start 0 --stop 4 --labels 6 --index 7 -t csv1 csv2 [..., csvN]

    Note:
    ----
    All flags are optional and default to the values above.
"""
import sys, os
import argparse

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


# =====
# Process the args, load and concatenate the data, do any preprocessing and 
# classify it, writing the results as name
# =====

# ----
# Process
# ----
parser = argparse.ArgumentParser(
        description="Run RFE using Linear SVMs on the supplied csv data")
parser.add_argument("name", 
        help="The name of the file to save the classfier results to")
parser.add_argument("--start", default=0, type=int, 
        help="Column number where features begin")
parser.add_argument("--stop", default=4, type=int, 
        help="Column number where features end")
parser.add_argument("--labels", default=6, type=int, 
        help="Column for labels")
parser.add_argument("--index", default=7, type=int, 
        help="Column number for cross_validation codes")
parser.add_argument("-t", nargs="+", 
        help="Targets, i.e. csv files to concatnate then classify")
args = parser.parse_args()

csvs = args.t
for csvn in csvs:
    if not os.path.exists(csvn):
        ## Yes neck beards I should open it 
        ## instead of testing for existence....
        raise IOError("{0} does not exist".format(csvn))

# ----
# Load
# ----
feature_index = range(args.start, args.stop+1)
X, y, index = None, None, None
for csvn in csvs:
    dftmp = pd.read_csv(csvn, sep=",", header=0)
    ## Reinit 
    if (X is None) and (y is None):
        X = np.asarray(dftmp.ix[:,feature_index])
        y = np.asarray(dftmp.ix[:,args.labels], dtype=np.str)
        index = np.asarray(dftmp.ix[:,args.index])
    ## Otherwise stack
    else:
        X = np.vstack([X, np.asarray(dftmp.ix[:,feature_index])])
        y = np.concatenate([y, np.asarray(dftmp.ix[:,args.labels], 
                dtype=np.str)])
        index = np.concatenate([index, np.asarray(dftmp.ix[:,args.index])])

# ----
# Preproces
# ----
# Convert y to integer codes
y = create_y(y)

# Sane so far?
assert checkX(X)
assert X.shape[0] == y.shape[0], "X and y length mismatch"

# Use trial chunks of labels to CV?
if args.labels == args.index:
    cv = KFold(y.shape[0], n_folds=5, indices=True)
else:
    # ----
    # To split by trial
    # Convert the trial index 
    # to trial/chunk counter
    #chunks = []
    #cnt = -1
    #for i in index:
    #    if i == 0:
    #        cnt += 1
    #    chunks.append(cnt)
    #chunks = np.asarray(chunks)
    #cv = KFoldChunks(chunks, 5, indices=True, min_size=15)
    # ----

    # ----
    # To split by subject
    unindex = sorted(np.unique(index))
    chunks = np.repeat(-1, index.shape[0])   
        ## Ensures no data are missed during recode.
    for k, ui in enumerate(unindex):
        chunks[ui == index] += np.int(k + 1)   ## +1 to adj for -1 init
    cv = LeavePLabelOut(chunks, p=3)
    # ----

    # Sane?
    assert X.shape[0] == chunks.shape[0], "X and chunks length mismatch"       
    assert np.sum(chunks == -1) == 0, "Chunks is malformed"  

# ----
# Classify
# ----

# ----
# Using RFE and linear SVM
clf = SVC(C=10, kernel="linear")
rfecv = RFECV(estimator=clf, step=1, cv=cv, scoring="accuracy")
rfecv.fit(X, y)
prediction = rfecv.predict(X)
print("Optimal feature number {0}".format(rfecv.n_features_))
print("Feature ranks {0}".format(rfecv.ranking_))
accs = accuracy_score(y, prediction)
#print(classification_report(y, prediction))
#print("Overall accuracy: {0}, Chance: {1}.".format(
#    np.round(accuracy_score(y, prediction), decimals=2), 
#    1.0/len(np.unique(y))))
# ----

# ----
# Using GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
#        max_depth=1, random_state=0)
#accs = cross_val_score(clf, X, y=y, scoring="accuracy", cv=cv,
#        n_jobs=1, verbose=0,
#        fit_params=None, score_func=None)

f = open("results.txt", "a")
f.write("{0},{1},{2}\n".format(    
    np.round(np.mean(accs), decimals=2), 
    np.round(1.0/len(np.unique(y)), decimals=2),
    args.name))
f.close()
# ----
