"""Misclanous utlity functions for these experiments."""
import os
import numpy as np
import pandas as pd

from copy import deepcopy

from fmrilearn.preprocess.labels import create_y
from fmrilearn.load import load_roifile
from fmrilearn.preprocess.data import checkX

from simfMRI.norm import zscore


def load_dimreduce_data_fromcl(csvs, feature_col, label_col, cv_col, trial_tr_col):
    """Load dimensionality reduced data (in csv files)
    
    Parameters
    ---------
    csvs : list([str, ...]))
        A list of csv file paths
    
    Returns
    ------
  
    """

    # Ensure the csvs exists
    for csvn in csvs:
        if not os.path.exists(csvn):
            raise IOError("{0} does not exist".format(csvn))

    Xs, ys, cvcodes, trs = [], [], [], [] 
    for csvn in csvs:
        dftmp = pd.read_csv(csvn, sep=",", header=0)
        Xs.append(np.asarray(dftmp.ix[:,feature_col]))
        ys.append(np.asarray(dftmp.ix[:,label_col], dtype=np.str))
        cvcodes.append(np.asarray(dftmp.ix[:,cv_col]))
        trs.append(np.asarray(dftmp.ix[:,trial_tr_col]))

    # Convert tr to a index for each trial
    cnt = 0
    last_ttr = 0
    indices = []
    for tr in trs:
        index = []
        for ttr in tr:
            # If the TR is not monotonic reset
            if ttr < last_ttr:  
                cnt += 1

            index.append(cnt)
            last_ttr = deepcopy(ttr)
        indices.append(index)

    # Convert to np.array
    for i in range(len(ys)):
        ys[i] = np.asarray(ys[i])
        indices[i] = np.asarray(indices[i])
        cvcodes[i] = np.asarray(cvcodes[i])

    return Xs, ys, indices, cvcodes


def extract_trial_features(X, trials, window, other=None, baseline=None):
    """Extract trial-level features from X
    
    Parameters
    ---------
    X : 2d array-like (column oriented)
        The raw features
    trials : 1d array-like
        A trial index
    window : list
        Index into each trial in trials
    other : list(np.array, ...)
         A list of other metadata to convert to
         the same space as trials, default is None.
    baseline : literal, optional
        Trial codes that indicate baseline trials, 
        or other things to ignore (defaults to None,
        which means all trials are used).

    Returns
    -------
    Xfea : 2d array-like (n_trials, n_col_X * 5)
        The extracted features, five per col in X
    trialsfea : 1d array-like (n_trials,)
        The new trial index
    otherfea : list(np.array, ...) (n_trials,)
         The new 'other' metadata
    """

    trialfea = []
    otherfea = None
    if other != None:
        otherfea = [[] for i in range(len(other))]
    
    # Number of rows in Xfea
    nrow = np.unique(trials).shape[0] 
    if baseline != None:
        if baseline not in np.unique(trials):
            raise ValueError("baseline ({0}) not in trials".format(baseline))
        nrow -= 1

    Xmax = np.zeros((nrow, X.shape[1]), dtype=np.float)
    Xmin = np.zeros_like(Xmax)
    Xdiff = np.zeros_like(Xmax)
    Xmean = np.zeros_like(Xmax)
    Xslope = np.zeros_like(Xmax)

    i = 0
    for trial in np.unique(trials):
        if (baseline != None) and (trial == baseline):
            continue

        # Locate this trials data
        mask = trial == trials
        x_trial = X[mask,:]

        # Extract features
        Xmax[i,:] = np.argmax(x_trial[window,], axis=0)
        Xmin[i,:] = np.argmin(x_trial[window,], axis=0)
        Xdiff[i,:] = Xmax[i,:] - Xmin[i,:]
        Xmean[i,:] = x_trial[window,].mean(axis=0)

        # Taken from Ploran 2007 definition
        # slope = (x_peak - x_onset) / (t_peak - t_onset)
        rise = np.max(x_trial[window,], axis=0) - x_trial[window,][0]
        run = Xmax[i,:] - 1
        run[run <= 0] = 1  ## Run can't be 0
        Xslope[i,:] = rise/run

        # Only need one (new) label for each trial
        trialfea.append(trials[mask][0])
        
        # Process other
        if other != None:
            for n in range(len(other)):
                otherfea[n].append(other[n][mask][0])
        i += 1

    # and join
    Xfea = np.hstack([Xmax, Xmin, Xdiff, Xmean, Xslope])
    #Xfea = np.hstack([Xmax, Xmin, Xdiff])
    
    # Recast otherfea to arrays
    if otherfea != None:
        for n in range(len(otherfea)):
            otherfea[n] = np.asarray(otherfea[n])
            
    return Xfea, np.asarray(trialfea), otherfea


def process_exp_argv(argv):
    """Process pos arguments for python programs with a invocation like:
            
            $ python ./script.py name data roifile cond [, filtfile]
    """
    if len(argv) == 6:
        basename = argv[1]  ## name
        dataname = argv[2]  ## data
        _, rois = load_roifile(argv[3])  ## roifile
        cond = argv[4]  ## cond
        tr = argv[5]
        filtfile = None
    elif len(argv) == 7:
        basename = argv[1]
        dataname = argv[2]
        _, rois = load_roifile(argv[3])
        cond = argv[4]
        tr = argv[5]
        filtfile = argv[6]
    else:
        raise ValueError("Wrong number of arguments")

    return basename, dataname, rois, cond, tr, filtfile


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

