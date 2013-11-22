"""EVA reduction for all voxels and trials.  Results saved to a series
of tables named like <name>_<roi>_<cond>.csv.

usage: python ./eva.py name data roifile cond [, filtfile]
"""
import sys, os
import numpy as np

from fmrilearn.load import load_roifile
from fmrilearn.analysis import fir

from fmrilearnexp.base import AverageTime
from fmrilearnexp.base import DecomposeExp
from wheelerdata.load.meta import get_data
from fmrilearnexp.common import process_exp_argv


# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
basename, dataname, rois, cond, filtfile = process_exp_argv(sys.argv)
data = get_data(dataname)

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = AverageTime(fir)
exp = DecomposeExp(spacetime, data, window=15, nsig=3)

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{0} ({1}/{2})".format(roi, n+1, len(rois)))   ## Progress marker
    exp.run(basename, roi, cond, smooth=False, filtfile=filtfile, event=True)


