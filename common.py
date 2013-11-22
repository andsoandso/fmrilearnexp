"""Misclanous utlity functions for these experiments."""
import os
import numpy as np

from fmrilearn.load import load_roifile
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.split import by_labels
from fmrilearn.preprocess.reshape import by_trial



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

