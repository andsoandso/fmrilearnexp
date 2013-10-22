"""PCA reduction for all voxels and trials.  Results saved to a series
of tables named like <name>_<param_name>_<cond>.csv.

usage: python ./pca_simspace.py name paramfile [, filtfile]
"""

import sys, os
import json
import numpy as np

from sklearn.decomposition import PCA

from fmrilearn.load import load_roifile
from fmrilearn.analysis import eva

from fmrilearnexp.base import Space
from fmrilearnexp.base import DecomposeSimulation


# ---------------------------------------------------------------------------
# Process argv
# ---------------------------------------------------------------------------
# pos: (1) name (2) paramfile (3) cond
# op: (4) filtfile
if len(sys.argv) == 3:
    basename = sys.argv[1]
    paramsets = json.load(open(sys.argv[2], "r"))
    filtfile = None
elif len(sys.argv) == 4:
    basename = sys.argv[1]
    paramsets = json.loads(open(sys.argv[2], "r"))
    filtfile = sys.argv[4]
else:
    raise ValueError("Wrong number of arguments")

# ---------------------------------------------------------------------------
# Setup exp
# ---------------------------------------------------------------------------
spacetime = Space(PCA(6, whiten=True), eva, mode="decompose")
exp = DecomposeSimulation(spacetime, window=11, nsig=3)

# ---------------------------------------------------------------------------
# And each sim parameter set
# ---------------------------------------------------------------------------
for n, paramsetname in enumerate(paramsets.keys()):
    print("{0} ({1}/{2})".format(paramsetname, n+1, len(paramsets)))   
    exp.run(basename+"_"+paramsetname, smooth=False, filtfile=filtfile, 
            **paramsets[paramsetname])

