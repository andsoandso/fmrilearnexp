"""usage: RT_corr roiname savename
"""
import os, sys
import numpy as np
import pandas as pd

from multiprocessing import Pool

from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, accuracy_score,
        precision_score, recall_score, f1_score)

from metaaccumulate.datatools.load import load_roi, create_y
from metaaccumulate.datatools.process import (remove_invariant_features,
        smooth)

# HISTORY
# * r1
# * r2 uses CSV C=10, rbf kernel, removed mean and var
def _save_accuracy_table(name, meta, acc):
    f = open(name,'a')
    
    # Convert meta if it's a seq
    if isinstance(meta, str):
        metastr = meta
    else:
        metastr = ",".join([str(m) for m in meta])
    
    f.write("{0},{1}\n".format(metastr, acc))
    f.close()


def _create_X_stats(X, trial_index, labels):
    trials = np.unique(trial_index)

    # ----
    # Init
    Xmax = np.zeros((trials.shape[0], X.shape[1]))
    Xmin = np.zeros_like(Xmax)
    #Xmean = np.zeros_like(Xmax)
    #Xvar = np.zeros_like(Xmax)

    # ----
    # Create the stats
    newlabels = []
    for ii, trial in enumerate(trials):
        # Locate this trials data
        mask = trial == trial_index
        x_trial = X[mask,:]

        # Get time to peak/min
        Xmax[ii,:] = np.argmax(x_trial, axis=0)
        Xmin[ii,:] = np.argmin(x_trial, axis=0)
        
        # And their diff
        Xdiff = Xmax - Xmin
        
        # Finally get trial means and variances
        #Xmean[ii,:] = x_trial.mean()
        #Xvar[ii,:] = x_trial.var()

        # Only need one label for each trial
        # now, the first is as good as any
        newlabels.append(labels[mask][0])
    
    Xfea = np.hstack([Xmax, Xmin, Xdiff])

    return Xfea, np.array(newlabels)


# ----
# OVERVIEW:
# * Load each Ss data, 
# * filter by TR, 
# * filter by resp, 
# * split into three folds, 
# * CV train/test manually, 
# * print the results
def exp(roi):
    accuracy_table_name = "rtstats_seperateresp_accuracy_r1.txt"

    roibasepath = "/data/data2/meta_accumulate/fh/roinii"
    roipaths = [
        os.path.join(roibasepath, roi+"_9.nii.gz"),
        os.path.join(roibasepath, roi+"_11.nii.gz"),
        os.path.join(roibasepath, roi+"_13.nii.gz"),
        os.path.join(roibasepath, roi+"_14.nii.gz"),
        os.path.join(roibasepath, roi+"_15.nii.gz"),
        os.path.join(roibasepath, roi+"_17.nii.gz"),
        os.path.join(roibasepath, roi+"_19.nii.gz"),
        os.path.join(roibasepath, roi+"_21.nii.gz"),
        os.path.join(roibasepath, roi+"_23.nii.gz"),
        os.path.join(roibasepath, roi+"_24.nii.gz"),
        os.path.join(roibasepath, roi+"_25.nii.gz"),
        os.path.join(roibasepath, roi+"_26.nii.gz"),
        os.path.join(roibasepath, roi+"_27.nii.gz"),
        os.path.join(roibasepath, roi+"_28.nii.gz")
        ]

    metabasepath = "/data/data2/meta_accumulate/fh/fidl"
    metapaths = [
        os.path.join(metabasepath, "trtime_fh009_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh011_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh013_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh014_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh015_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh017_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh019_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh021_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh023_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh024_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh025_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh026_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh027_RT_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh028_RT_corr_EF.csv")
    ]
    
    metapaths2 = [
        os.path.join(metabasepath, "trtime_fh009_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh011_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh013_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh014_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh015_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh017_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh019_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh021_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh023_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh024_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh025_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh026_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh027_motor_EF.csv"),
        os.path.join(metabasepath, "trtime_fh028_motor_EF.csv")
    ]

    # ----
    # Loop overs Ss data, preprocess, and classify
    overall_accuracy = []
    for roipath, metapath, metapath2 in zip(roipaths, metapaths, metapaths2):
        print("----")
        print("Loading data.")
        print("\t{0}".format(roipath))
        print("\t{0}".format(metapath))
    
        _, roiname = os.path.split(roipath)
                ## USe roiname in the accuracy tables

        # ----
        # Load and process labels

        # --
        # Load for metapaths
        meta = pd.read_csv(metapath)    
        meta2 = pd.read_csv(metapath2)

        print("Before filteration.")
        trs = np.array(meta["TR"].tolist())
        print("\tNumber of trs: {0}".format(trs.shape))

        reaction_times = np.array(meta["rt"].tolist())
        print("\tNumber of reaction_times: {0}".format(
                reaction_times.shape))
        
        trial_index = np.array(meta["trialcount"].tolist())
        print("\tNumber of trial_index: {0}".format(
                trial_index.shape))

        # --
        # Load for metapath2
        resps2 = np.array(meta2["resp"].tolist())
        print("\tNumber of resps2: {0}".format(resps2.shape))
        
        trs2 = np.array(meta2["TR"].tolist())
        print("\tNumber of trs2: {0}".format(trs2.shape))

        # --
        # Find common trs
        common_trs = np.array([tr for tr in trs if tr in trs2])
        
        # --
        # Filter for meta
        inmeta = np.array([i for i, tr in enumerate(trs) if 
                tr in common_trs])        
        reaction_times = reaction_times[inmeta]
        trial_index = trial_index[inmeta]

        # --
        # Filter for meta2
        inmeta2 = np.array([i for i, tr2 in enumerate(trs2) if 
                tr2 in common_trs])
        resps2 = resps2[inmeta2]

        # --
        # Load the roi data
        Xsp = load_roi(roipath, sparse=True)
        print("\tXsp shape: {0}".format(Xsp.shape))

        # ----
        # Filtration
        Xsp = remove_invariant_features(Xsp, sparse=True)
        Xsp = Xsp[common_trs,:]

        rt_only = (reaction_times == "rt2") | (reaction_times == "rt3") 
        rt_only = np.arange(reaction_times.shape[0])[rt_only]
            ## Convert bool mask to int for sparse array compatiblity
        
        reaction_times = reaction_times[rt_only]
        resps2 = resps2[rt_only]
        trial_index = trial_index[rt_only]

        Xsp = Xsp[rt_only,:]

        print("After filteration.")
        print("\tNumber of trs: {0}".format(trs.shape))
        print("\tNumber of trs2: {0}".format(trs2.shape))

        print("\tNumber of reaction_times: {0}".format(
                reaction_times.shape))
        print("\tNumber of trial_index: {0}".format(
                trial_index.shape))
        print("\tNumber of resps2: {0}".format(resps2.shape))

        print("\tXsp shape: {0}".format(Xsp.shape))
        
        # --
        # Split into resp
        is_left = np.arange(resps2.shape[0])[resps2 == "left"]
        is_right = np.arange(resps2.shape[0])[resps2 == "right"] 
        select_lr = {"left" : is_left, "right" : is_right}
        for name, index in select_lr.items():
            print("**{0}**".format(name))

            Xsp_sel = Xsp[index,:]
            reaction_times_sel = reaction_times[index]
            trial_index_sel = trial_index[index]

            # ----
            # Smooth (and desparse)
            X = smooth(Xsp_sel.todense(), tr=1.5, ub=0.06, lb=0.001)

            # ----
            # Create stats
            X, reaction_times_sel = _create_X_stats(X, trial_index_sel,
                    reaction_times_sel)

            print("After smooth and stat creation.")
            print("\tNumber of trials: {0}".format(
                    np.unique(trial_index_sel).shape))
            print("\tNumber of reaction_times_sel: {0}".format(
                    reaction_times_sel.shape))
            print("\tX shape: {0}".format(X.shape))
            
            print("Label counts:")
            for rt in np.unique(reaction_times_sel):
                print("\t({0}, {1})".format(rt, np.sum(
                    reaction_times_sel == rt)))

            # ----
            # Create train/class paritions
            nrow, ncol = X.shape
            cv = KFold(nrow, n_folds=5, indices=True)
            
            # ----
            # Loop over the folds, class as we go
            print("Classifying.")
            i = 0
            y = create_y(reaction_times_sel)
            for train_index, test_index in cv:
                # ----
                # Partition the data
                Xtrain = X[train_index,:]
                Xtest = X[test_index,:]
                print("\tShapes of Xtrain and Xtest: {0}, {1}".format(
                        Xtrain.shape, Xtest.shape))
                
                ytrain = y[train_index]
                ytest = y[test_index]
                print("\tNumber of ytrain and ytest: {0}, {1}".format(
                        ytrain.shape, ytest.shape))

                # ----
                # Init clf
                clf = GradientBoostingClassifier(
                        n_estimators=100, learning_rate=1.0, 
                        max_depth=1, random_state=0)
                #clf = SVC(C=10, kernel='rbf')

                # ----
                # Class!
                clf.fit(scale(Xtrain), ytrain)
                predictions = clf.predict(scale(Xtest))
                
                # ----
                # And print the results
                accuracy = accuracy_score(ytest, predictions) 
                print("\tFor CV {0}, accuracy was {1}".format(
                        i, accuracy))        
                
                # Store the accuracy
                overall_accuracy.append(accuracy)
                
                print("\tSaving accuracy")
                _save_accuracy_table(
                    accuracy_table_name, 
                    [name+"_"+roi, name+"_"+roiname, "cv_"+str(i)], 
                    accuracy)
                
                # ----
                # Update the Cv counter
                i += 1
    
    print("****")
    omean = np.array(overall_accuracy).mean()
    print("Overall accuracy for {1} was {0}".format(omean, roi))

    print("Saving accuracy")
    _save_accuracy_table(accuracy_table_name, 
            [roi, roi, "overall"], 
            omean)


if __name__ == "__main__":
    rois = ["respXtime_rfx_mask",
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

    [exp(roi) for roi in rois]

