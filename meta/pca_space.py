"""PCA reduction for all voxels and trials.  Results saved to a series
of tables named like <name>_<roi>_<cond>.csv.

usage: python ./pca_space.py name data roifile cond [, filtfile]
"""

import sys, os
import numpy as np

from sklearn.decomposition import PCA

from fmrilearn.load import load_roifile
from fmrilearn.analysis import fir

from fmrilearnexp.base import Space
from fmrilearnexp.base import DecomposeExp

from wheelerdata.load.butterfly import Butterfly
from wheelerdata.load.clock import Clock
from wheelerdata.load.fh import FH
from wheelerdata.load.polygon import Polygon
from wheelerdata.load.redgreen import Redgreen


# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
if len(sys.argv) == 4:
    basename = sys.argv[1]  ## name
    dataname = sys.argv[2]  ## data
    _, rois = load_roifile(sys.argv[3])  ## roifile
    cond = sys.argv[4]  ## cond
    filtfile = None
elif len(sys.argv) == 6:
    basename = sys.argv[1]
    dataname = sys.argv[2]
    _, rois = load_roifile(sys.argv[3])
    cond = sys.argv[4]
    filtfile = sys.argv[5]
else:
    raise ValueError("Wrong number of arguments")

# ---------------------------------------------------------------------------
# Process exp
# ---------------------------------------------------------------------------
if dataname == 'fh':
    data = FH()
elif dataname == 'butterfly':
    data = Butterfly()
elif dataname == 'clock':
    data = Clock()
elif dataname == 'polygon':
    data = Polygon()
elif dataname == 'redgreen':
    data = Redgreen()
else:
    raise ValueError('Data not understood.  Try fh, butterfly, clock, 
            polygon, or redgreen')

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = Space(PCA(6, whiten=True), fir, mode="decompose")
exp = DecomposeExp(spacetime, data, window=15, nsig=3)

# ---------------------------------------------------------------------------
# And run each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{0} ({1}/{2})".format(roi, n+1, len(rois)))
    exp.run(basename, roi, cond, smooth=False, filtfile=filtfile, event=True)

