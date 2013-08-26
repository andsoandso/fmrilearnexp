""" See exp() for details:

Usage
-----
* Command line:
    python ./motor_stats_between.py   ## Runs many ROIs (runtime is ~10 hours)

* ipython:
    >>> from fmrilearnexp.fh.motor_stats import exp
    >>> # Use the known good functional ROI....
    >>> exp("respXtime_rfx_mask",
            "/data/data2/meta_accumulate/fh/roinii",
            "/data/data2/meta_accumulate/fh/fidl",
            "fh_motor_stats_between_accuaracy_test.txt"
            True)
"""
import os
import numpy as np
import pandas as pd

from functools import partial

# sklearn imports.
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score,
        precision_score, recall_score, f1_score)

# fmrilean imports.
from fmrilearn.load import load_nii
from fmrilearn.save import (save_accuracy_table, reset_accuracy_table)
from fmrilearn.preprocess.data import (remove_invariant_features, 
        create_X_stats, smooth)
from fmrilearn.preprocess.labels import (construct_targets, 
        construct_filter, filter_targets)
from fmrilearn.info import (print_target_info, print_X_info, print_clf_info,
        print_label_counts)
from fmrilearn.classify import simple

# wheelerdata import
from wheelerdata.load.fh import (get_roi_data_paths,
        get_motor_metadata_paths, get_subject_dirs)


def _save_preprocessed_csv(name, X, labels, labels_name):
    df = pd.DataFrame(X)
    df[labels_name] = labels
    df.to_csv(name, index=False)


def _load_preprocessed_csv(name, labels_name):
    df = pd.read_csv(name)
    labels = np.array(df[labels_name].tolist())
    df = df.drop(labels_name, axis=1)
    X = np.array(df.as_matrix())
    
    return X, labels


def preprocess(roi, verbose=True):
    roipaths = get_roi_data_paths(roi)
    metapaths = get_motor_metadata_paths()
    
    # ----
    # Loop overs Ss data, preprocess, and classify
    for roipath, metapath in zip(roipaths, metapaths):
        if verbose:
            print("----")
            print("Preprocessing.")
            
            print("Loading data.")
            print("\t{0}".format(roipath))
            print("\t{0}".format(metapath))
   
        # Drop path and extension
        _, roiname = os.path.split(roipath)

        # ----
        # Load and process labels
        meta = pd.read_csv(metapath)
        resps = np.array(meta["resp"].tolist())
        trial_index = np.array(meta["trialcount"].tolist())
        trs = np.array(meta["TR"].tolist())
    
        # ----
        # Construct targets
        targets = construct_targets(resps=resps, 
                trial_index=trial_index,
                trs=trs)
    
        # ----
        # Get the data, intial preprocessing
        Xsp = load_nii(roipath, sparse=True)
        Xsp = remove_invariant_features(Xsp, sparse=True)
        Xsp = Xsp[targets["trs"],:]

        if verbose:
            print("Before filtration.")
            print_target_info(targets)
            print_X_info(Xsp)

        # --
        # Filter...
        # --
        # By rt
        keepers = ["left", "right"]
        keep_lr = construct_filter(targets["resps"], keepers, True)
        targets = filter_targets(keep_lr, targets)
        Xsp = Xsp[keep_lr,:]
        
        if verbose:
            print("After filtration.")
            print_target_info(targets)
            print_X_info(Xsp)
        
        # --
        # Smooth (and desparse)
        X = smooth(Xsp.todense(), tr=1.5, ub=0.06, lb=0.001)
        
        # --
        # Create stats
        Xstat, resps_stat = create_X_stats(
                X,
                targets["trial_index"], 
                targets["resps"])
        
        if verbose:
            print("After statistics creation.")
            print("\tNumber of resps_stat: {0}".format(resps_stat.shape))
            print_X_info(Xstat)

            print("Saving data.")

        _save_preprocessed_csv("{0}.csv".format(roiname),
                Xstat, resps_stat, "resps_stat")


def exp(roi, table, verbose):    
    """
    An experimental function to classify all permutations of subject data
    pairs for the ROI from the face/house data by motor response.
    
    For example, if we had only s1, s2, and s3 this experiment 
    would train on s1 and test on s2 and s3, i.e.
        * (train,test): (s1,s2), (s1,s3)
    then 
        * (train,test): (s2,s1), (s2,s3)
    and finally
        * (train,test): (s3,s2), (s3,s1)
    
    storing each iteration in <table>.  Note that train test sets are not 
    symmetric, i.e., (s1,s2) is not the same as (s2,s1)
    
    Parameters
    ----------
    roi - the name of the roi to use
    roipath - the full path to the nifti data to classfiy
    metapath - the full path to the metadata (labels)
    table - the name of results table.
    verbose - Print debugging/status info (True).  If False this 
        function is silent.
    
    Return
    ------
    Saves a table of classification accuracies for each subject's 
        cross validation fold (n=5) as well as the overall subject 
        mean.
    
    Data
    ----
        * Face/house
        * Note: runs each subject in the face/house set in a separate
            classification experiment.

    Details
    -------
        * Classes:
            1. 'resp', i.e. motor response - 'left'/'right'

        * Preprocessing:
            1. remove_invariant_features, 
            2. smooth, 
            3. create_X_stats
    """
    
    # Load a Ss preprocessed data, train on it
    # Then test against the rest of the Ss,
    # loading as needed.
    roipaths = get_roi_data_paths(roi)
    subjectnames = get_subject_dirs()
        ## Use as subject names in accuracy table

    # --
    # clf setup
    clf = GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, 
            max_depth=1, random_state=0)
    
    # --
    if verbose:
        print("----")
        print("Classifying...")
        print_clf_info(clf)
 
    # ----
    # Compare between subject classifier generality.
    # --
    # FIrst get this itrations (Ss) data and ...
    for roipath, s in zip(roipaths, subjectnames):
        if verbose:
            print("Loading data.")
            print("\t{0}".format(roipath))
 
        _, roiname = os.path.split(roipath)
        Xstat, resps_stat = _load_preprocessed_csv(
                "{0}.csv".format(roiname), "resps_stat")
       
        if verbose:
            print("After loading.")
            print("\tNumber of resps_stat: {0}".format(resps_stat.shape))
            print_X_info(Xstat)
 
        # --
        # compare to the rest
        truths = []
        predictions = []
        s_inners = []
        for roipath_inner, s_inner in zip(roipaths, subjectnames):
            if s == s_inner: 
                continue
            
            # --
            # Load the inner data
            _, roiname_inner = os.path.split(roipath_inner)
            Xstat_inner, resps_stat_inner = _load_preprocessed_csv(
                    "{0}.csv".format(roiname_inner), "resps_stat")

            if verbose:
                print("After inner loading:")
                print("\tNumber of resps_stat_inner: {0}".format(
                        resps_stat_inner.shape))
                print_X_info(Xstat_inner)

            # -- Make sure Xs have the same number of cols
            if Xstat.shape[1] != Xstat_inner.shape[1]:
                print("***WARNING.***")
                print("{0} and {1} didn't match.".format(
                        s, s_inner))
                print("***SKIPPING {0}.***".format(s_inner))
                continue

            # --
            # Go!
            truth, prediction = simple(
                    Xstat, Xstat_inner, 
                    resps_stat, resps_stat_inner, 
                    clf,
                    verbose)
            
            truths.append(truth)
            predictions.append(prediction)
            s_inners.append(s_inner)
        
        # ----
        # Save the results
        for truth, prediction, s_inner in zip(
                truths, predictions, s_inners):
            accuracy = accuracy_score(truth, prediction)
            save_accuracy_table(
                table, 
                [roi, roiname, "{0}_{1}".format(s,s_inner)], 
                accuracy)
            
            if verbose:
                print("Processing the results...")
                print("\tBetween {0} and {1}, accuracy was {2}".format(
                        s, s_inner, accuracy))
                print("\tFull report:")
                print(classification_report(truth, prediction, 
                        target_names=sorted(np.unique(resps_stat))))


if __name__ == "__main__":
    # ----
    # Experimental init
    
    # --
    # Exp globals
    VERBOSE = True
    ACCURACY_TABLE_NAME = "fh_motor_stats_between_accuracy.txt"
    ROIS = ["respXtime_rfx_mask",
           "left_ventrical",
           "right_ventrical",
           "left_putamen",
           "right_putamen",
           "left_caudate",
           "right_caudate",
           "sma",
           "precentral",
           "postcentral",
           "parietal_superior",
           "loc_superior",
           "loc_iferior",
           "mfg",
           "sfg",
           "insula",
           "ifg_triangularis",
           "ifg_opercularis",
           "stempotal_anterior",
           "stempotal_posterior",
           "acc",
           "pcc",
           "precuneous",
           "ofc",
           "left_hippocampus",
           "right_hippocampus",
           "parahippo_anterior",
           "parahippo_posterior"]

    # ROIS = ["respXtime_rfx_mask",]  ## Test/Debug

    # --     
    # Pre-run cleanup
    reset_accuracy_table(ACCURACY_TABLE_NAME)
       
    # --
    # A little PFA to simplify the run loops...
    pfa_preprocess = partial(preprocess, verbose=VERBOSE)
    pfa_exp = partial(exp, table=ACCURACY_TABLE_NAME, verbose=VERBOSE)
    
    # ----
    # And go!
    for roi in ROIS:
        pfa_preprocess(roi)
        pfa_exp(roi)
