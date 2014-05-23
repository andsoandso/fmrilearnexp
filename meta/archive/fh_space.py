"""
usage: python ./fh_space.py name data roifile cond [, filtfile]
"""

import sys, os
import numpy as np

from fmrilearn.analysis import fir

from sklearn.decomposition import FactorAnalysis

from fmrilearnexp.base import Space
from fmrilearnexp.base import DecomposeExp
from fmrilearnexp.common import process_exp_argv

from wheelerdata.load.meta import get_data

# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
basename, dataname, rois, cond, filtfile = process_exp_argv(sys.argv)
data = get_data(dataname)

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = Space(FactorAnalysis(5), fir, mode="decompose")
exp = DecomposeExp(spacetime, data, window=15, nsig=3)

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{3}: {0} ({1}/{2})".format(roi, n+1, len(rois), dataname))   
    exp.run(basename, roi, cond, smooth=False, filtfile=filtfile, event=True)

