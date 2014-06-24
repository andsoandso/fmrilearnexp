"""
usage: python ./kmeans_tc.py name data roifile cond tr window [, filtfile]
"""

import sys, os
import numpy as np
import argparse

from fmrilearn.load import load_roifile
from sklearn.decomposition import FastICA
from wheelerexp.base import SelectTimecourse
from wheelerexp.base import DecomposeExp
from wheelerdata.load.meta import get_data

# from fmrilearn.analysis import fir
from fmrilearn.load import load_roifile
from wheelerdata.load.meta import get_data

parser = argparse.ArgumentParser(
        description="Apply PCA to trial-level data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
parser.add_argument(
        "name", 
        help="Name of this exp"
        )
parser.add_argument(
        "data", 
        help="Name of the wheeerlab dataset"
        )
parser.add_argument(
        "roifile", 
        help="A text file with ROI names to iterate over, ':' separated"
        )
parser.add_argument(
        "cond", 
        help="Name of cond to use"
        )
parser.add_argument(
        "tr", 
        help="TR of the dataset",
        type=float
        )
parser.add_argument(
        "window", 
        help="L of the trial window (in TRs)",
        type=int
        )
parser.add_argument(
        "filtfile", 
        help="Name of this exp",
        nargs='?',
        default=None
        )
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
# Replace this with good arg processing....
# basename, dataname, rois, cond, tr, filtfile = process_exp_argv(sys.argv)
data = get_data(args.data)
_, rois = load_roifile(args.roifile)  ## roifile

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = SelectTimecourse(FastICA(6, max_iter=500), mode="decompose")
exp = DecomposeExp(
        spacetime, data, window=args.window, nsig=3, tr=args.tr
        )

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{3}: {0} ({1}/{2})".format(roi, n+1, len(rois), args.data))   
    exp.run(
            args.name, roi, args.cond, smooth=False, 
            filtfile=args.filtfile, event=False
            )

