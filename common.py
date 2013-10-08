"""Misclanous utlity functions for these experiments."""
import os
import numpy as np
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.split import by_labels
from fmrilearn.preprocess.reshape import by_trial


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

