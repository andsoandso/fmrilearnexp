"""PCA reduction for all voxels and trials.  Results saved to a series
of tables named like <name>_<roi>_<cond>.csv.

usage: python ./pca_spacetime.py name roifile cond [, filtfile]
"""
import sys, os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA

from fmrilearn.load import load_meta
from fmrilearn.load import load_nii
from fmrilearn.load import load_roifile
from fmrilearn.save import save_tcdf
from fmrilearn.preprocess.data import filterX
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.labels import construct_targets
from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets

from wheelerdata.load.fh import get_roi_data_paths
from wheelerdata.load.fh import get_metapaths_containing
from wheelerdata.load.fh import get_motor_metadata_paths

from fmrilearnexp.base import Spacetime
from fmrilearnexp.base import DecomposeFH
from fmrilearnexp.common import get_roiname
from fmrilearnexp.common import join_by_underscore

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
spacetime = Spacetime(PCA(6, whiten=True))
exp = DecomposeFH(spacetime, window=11, nsig=3)

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{0} ({1}/{2})".format(roi, n+1, len(rois)))   ## Progress marker
    exp.run(basename, roi, cond, smooth=False, filtfile=None)
