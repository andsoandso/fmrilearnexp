"""Misclanous utlity functions for these experiments."""
import os
import numpy as np
import pandas as pd

from copy import deepcopy

from fmrilearn.preprocess.labels import create_y
from fmrilearn.load import load_roifile
from fmrilearn.preprocess.data import checkX


def process_exp_argv(argv):
    """Process pos arguments for python programs with a invocation like:
            
            $ python ./script.py name data roifile cond [, filtfile]
    """
    if len(argv) == 4:
        basename = argv[1]  ## name
        dataname = argv[2]  ## data
        _, rois = load_roifile(argv[3])  ## roifile
        cond = argv[4]  ## cond
        filtfile = None
    elif len(argv) == 6:
        basename = argv[1]
        dataname = argv[2]
        _, rois = load_roifile(argv[3])
        cond = argv[4]
        filtfile = argv[5]
    else:
        raise ValueError("Wrong number of arguments")

    return basename, dataname, rois, cond, filtfile


def get_roiname(roipath):    
    """Strip off paths and extensions"""

    _, roiname = os.path.split(roipath)
    roiname, _ = os.path.splitext(roiname)
    roiname, _ = os.path.splitext(roiname)

    return roiname


def join_by_underscore(append_csv, *args):
    tablename = "_".join([str(arg) for arg in args])
    if append_csv: tablename += ".csv"

    return tablename


def sum_explained_variance(pcar):
    return np.round(np.sum(pcar.explained_variance_ratio_), decimals=2)


def _create_X_y_xcs(csvs, data, labels_j, trial_tr_j, window):
    # Check data exists
    feature_index = range(*[int(i) for i in data.split(':')])
    for csvn in csvs:
        if not os.path.exists(csvn):
            raise IOError("{0} does not exist".format(csvn))

    X, y, index, tTR = None, None, None, None
    for csvn in csvs:
        dftmp = pd.read_csv(csvn, sep=",", header=0)

        ## Reinit 
        if (X is None) and (y is None):
            X = np.asarray(dftmp.ix[:,feature_index])
            y = np.asarray(dftmp.ix[:,labels_j], dtype=np.str)
            tTR = np.asarray(dftmp.ix[:,trial_tr_j])
        ## Otherwise stack
        else:
            X = np.vstack([X, np.asarray(dftmp.ix[:,feature_index])])
            y = np.concatenate([y, np.asarray(dftmp.ix[:,labels_j], 
                    dtype=np.str)])
            tTR = np.concatenate([tTR, np.asarray(dftmp.ix[:,trial_tr_j])])

    # Sane so far?
    assert checkX(X)
    assert X.shape[0] == y.shape[0], "X and y length mismatch"

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
        ## If the TR is not monotonic, new trial
        if ttr < last_ttr:  
            cnt += 1
        trial_index.append(cnt)
        last_ttr = deepcopy(ttr)

    # ----
    # Convert X to Xstat
    # --
    # Init
    trials = np.unique(trial_index)
    Xmax = np.zeros((trials.shape[0], X.shape[1]))
    Xmin = np.zeros_like(Xmax)
    Xmean = np.zeros_like(Xmax)
    Xvar = np.zeros_like(Xmax)
    ystat = []
    #indexstat = []
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
        Xmean[ii,:] = x_trial[window,].mean(axis=0)
        Xvar[ii,:] = x_trial[window,].var(axis=0)
           
        ystat.append(y[mask][0])
        #indexstat.append(index[mask][0])

    X = np.hstack([Xmax, Xmin, Xdiff, Xmean, Xvar])
    y = np.asarray(ystat)
    #index = np.asarray(indexstat)
    del Xmax, Xmin, Xdiff, Xmean, Xvar ## Cleanup ASAP

    # Still sane?
    assert checkX(X)
    assert X.shape[0] == y.shape[0], "X and y length mismatch"

    return X, y
