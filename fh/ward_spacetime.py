"""Kmeans clusters for all voxels and trials.  Results saved to a series
of tables named like <name>_<roi>_<cond>.csv.

usage: python ./ward_spacetime.py name roifile cond [, filtfile]
"""
import sys, os
import numpy as np

from sklearn.cluster import Ward
from fmrilearn.analysis import eva
from fmrilearn.load import load_roifile

from fmrilearnexp.base import Spacetime
from fmrilearnexp.base import DecomposeFH


# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
# pos: (1) name (2) roifile (3) cond
# op: (4) filtfile
if len(sys.argv) == 4:
    basename = sys.argv[1]
    _, rois = load_roifile(sys.argv[2])
    cond = sys.argv[3]
    filtfile = None
elif len(sys.argv) == 4:
    basename = sys.argv[1]
    _, rois = load_roifile(sys.argv[2])
    cond = sys.argv[3]
    filtfile = sys.argv[4]
else:
    raise ValueError("Wrong number of arguments")

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = Spacetime(Ward(n_clusters=6), mode="cluster")
exp = DecomposeFH(spacetime, window=11, nsig=3)

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{0} ({1}/{2})".format(roi, n+1, len(rois)))   ## Progress marker
    exp.run(basename, roi, cond, smooth=False, filtfile=None)
