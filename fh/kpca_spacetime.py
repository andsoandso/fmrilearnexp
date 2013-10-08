"""PCA reduction for all voxels and trials.  Results saved to a series
of tables named like <name>_<roi>_<cond>.csv.

usage: python ./kpca_spacetime.py name roifile cond [, filtfile]
"""
import sys, os
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

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
from fmrilearnexp.common import get_roiname
from fmrilearnexp.common import join_by_underscore

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
# PCA setup
kernel = "rbf"
spacetime = Spacetime(KernelPCA(6, kernel=kernel))

# X params
smooth = False
window = 11

# Write mode
nsig = 3  ## Sig fig on write
roicount = 0
mode = 'w'
header = True

# ---------------------------------------------------------------------------
# Pos args
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
    raise 

# ---------------------------------------------------------------------------
# PCA each roi
# ---------------------------------------------------------------------------
for n, roi in enumerate(rois):
    print("{0} ({1}/{2})".format(roi, n+1, len(rois)))   ## Progress marker
    table = join_by_underscore(False, basename, kernel, roi, cond)

    # Find the data paths
    roipaths = get_roi_data_paths(roi)
    metapaths = get_metapaths_containing(cond)

    # Reinit write flags
    roicount = 0
    mode = 'w'
    header = True
    
    # -----------------------------------------------------------------------
    # PCA each subject
    # -----------------------------------------------------------------------
    for roipath, metapath in zip(roipaths, metapaths):
        roiname = get_roiname(roipath)
        print("\t{0}".format(roiname))  ## Mark progress

        if roicount > 0:  
            mode = 'a'
            header = False    

        # Get metadata
        trs, trial_index, y = load_meta(["TR", "trialcount", cond], metapath)
        targets = construct_targets(trial_index=trial_index, trs=trs, y=y)

        # and data, preprocess too,
        X = load_nii(roipath, clean=True, sparse=False, smooth=smooth)
        X = X[trs,:]

        if filtfile is not None:
            filterX(filtfile, X, targets)

        # finally decompose.
        Xcs, csnames = spacetime.fit_transform(
                X, targets["y"], targets["trial_index"], window, norm=True)
        
        # Name them,
        csnames = ["all", ] + sorted(np.unique(targets["y"]))
        
        # and write.
        for Xc, csname in zip(Xcs, csnames):
            save_tcdf(
                    name=join_by_underscore(True, table, csname), 
                    X=Xc, 
                    cond=csname,
                    dataname=join_by_underscore(False, roiname, 
                            spacetime.decompr.n_components),
                    index='auto',
                    header=header, 
                    mode=mode,
                    float_format="%.{0}f".format(nsig))
        roicount += 1
