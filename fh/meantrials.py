"""Calculate average trials

usage: meantrials roi metapath"""
import sys, os

import pandas as pd
import numpy as np

from fmrilearn.analysis import mean_trial, fir_trial
from fmrilearn.preprocess.labels import (construct_targets, 
        construct_filter, filter_targets)
from fmrilearn.preprocess.data import remove_invariant_features
from fmrilearn.load import load_nii


def trial_estimates(roi, metapath)
    # ----
    # Globals
    TR = 1.5
    WINDOW = 28

    # ----
    # Load labels and construct targets
    meta = pd.read_csv(metapath)    

    resps = np.array(meta["resp"].tolist())
    trial_index = np.array(meta("trialcount").tolist())
    trs = np.array(meta["TR"].tolist())

    targets = construct_targets(
            resps=resps, 
            trial_index=trial_index,
            trs=trs)

    # ----
    # Get data, only keep labeled data, that which we have a TR for
    X = remove_invariant_features(load_nii(roi, sparse=True)).todense()
    X = X[targets["trs"],:]

    # ----
    # Filter for valid resp only
    keepers = ["left", "right"]
    keep_lr = construct_filter(targets["resps"], keepers, True)
    targets = filter_targets(keep_lr, targets)
    X = X[keep,:]

    # ----
    # Calc trial estimates
    Xmean, ymean = mean_trial(
                X, 
                targets["resps"], 
                targets["trial_index"])
    Xfir, yfir = fir_trial(
                X, 
                targets["resps"], 
                TR, WINDOW)
    Xfir_meanfirst, _ = firtrial(
                X.mean(1).reshape((X.shape[0],1)), 
                targets["resps"], 
                TR, WINDOW)

    # ----
    # And save them by first creating an informative prefix/basename
    path, roiname = os.path.split(roi)
    nifitname, ext = os.path.splitext(roiname)
    basename = "{0}_{1}".format(nifitname, cond)

    # Save X colmean with labels
    X_as_df = pd.DataFrame(X.mean(1))
    X_as_df['resps'] = targets["resps"]
    X_as_df['trial_index'] = trial_index
    X_as_df.to_csv('{0}_X_col_mean.csv'.format(basename), sep=",")

    # then saving in a R data.frame compatible format.
    df_mean = pd.DataFrame(Xmean)
    df_mean['ymean'] = ymean
    df_mean.to_csv('{0}_Xmean.csv'.format(basename), sep=',')

    df_fir = pd.DataFrame(Xfir)
    df_fir['yfir'] = yfir
    df_fir.to_csv('{0}_Xfir.csv'.format(basename), sep=',')

    df_fir_meanfirst = pd.DataFrame(Xfir_meanfirst)
    df_fir_meanfirst['yfir'] = yfir
    df_fir_meanfirst.to_csv(
            '{0}_Xfir_meanfirst.csv'.format(basename), sep=',')


if __name__ == "__est_trials__":

    #ROIS = ["respXtime_rfx_mask",
    #        "left_ventrical",
    #        "right_ventrical",
    #        "left_putamen",
    #        "right_putamen",
    #        "left_caudate",
    #        "right_caudate",
    #        "sma",
    #        "precentral",
    #        "postcentral",
    #        "parietal_superior",
    #        "loc_superior",
    #        "loc_iferior",
    #        "mfg",
    #        "sfg",
    #        "insula",
    #        "ifg_triangularis",
    #        "ifg_opercularis",
    #        "stempotal_anterior",
    #        "stempotal_posterior",
    #        "acc",
    #        "pcc",
    #        "precuneous",
    #        "ofc",
    #        "left_hippocampus",
    #        "right_hippocampus",
    #        "parahippo_anterior",
    #        "parahippo_posterior"]

    ROIS = ["respXtime_rfx_mask",]  ## Test/Debug
    for roi in ROIS:
        trial_estimates(roi, metapath)
