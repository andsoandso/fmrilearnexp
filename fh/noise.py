"""usage: motor roiname savename
"""
import os, sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, accuracy_score,
        precision_score, recall_score, f1_score)

from metaaccumulate.datatools.load import load_roi, create_y
from metaaccumulate.datatools.process import (remove_invariant_features,
        smooth)


# HISTORY
# * First go is r1
# * r2 uses only extreme noise value (66 v 70)
# * r3 is a rerun of r2 using two parahippo ROIS
def _save_accuracy_table(name, meta, acc):
    f = open(name,'a')
    
    # Convert meta if it's a seq
    if isinstance(meta, str):
        metastr = meta
    else:
        metastr = ",".join([str(m) for m in meta])
    
    f.write("{0},{1}\n".format(metastr, acc))
    f.close()


def _shiftby(X, labels, by):
    """Accounts for HRF lag. Shift X (a single array) and targets 
    (a list of ys and labs) by <by>. """
    
    by = int(by)
    
    # Do nothing when by is 0,
    if by == 0:
        return X, labels
    
    # otherwise shift by
    X = X[by:,:]  
    labels = labels[0:(labels.shape[0] - by)] 

    return X, labels


# ----
# OVERVIEW:
# * Load each Ss data, 
# * filter by TR, 
# * filter by resp, 
# * split into three folds, 
# * CV train/test manually, 
# * print the results
def main(roi):

    acc_table_name = "noise_accuracy_r2.txt"

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
        os.path.join(metabasepath, "trtime_fh009_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh011_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh013_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh014_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh015_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh017_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh019_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh021_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh023_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh024_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh025_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh026_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh027_noise_corr_EF.csv"),
        os.path.join(metabasepath, "trtime_fh028_noise_corr_EF.csv")
    ]

    # ----
    # Loop overs Ss data, preprocess, and classify
    overall_accuracy = []
    for roipath, metapath in zip(roipaths, metapaths):
        print("----")
        print("Loading data.")
        print("\t{0}".format(roipath))
        print("\t{0}".format(metapath))
    
        _, roiname = os.path.split(roipath)
                ## USe roiname in the accuracy tables

        # ----
        # Load and process labels
        meta = pd.read_csv(metapath)    
        
        print("Before filteration.")
        trs = np.array(meta["TR"].tolist())
        print("\tNumber of TRs: {0}".format(trs.shape))

        noises = np.array(meta["noise"].tolist())
        print("\tNumber of noises: {0}".format(noises.shape))

        Xsp = load_roi(roipath, sparse=True)
        print("\tXsp shape: {0}".format(Xsp.shape))

        # ----
        # Filtration
        Xsp = remove_invariant_features(Xsp, sparse=True)
        Xsp = Xsp[trs,:]

        #keepers = ["66percent", "67percent", "69percent", "68percent",
        #        "65percent", "70percent"]
        keepers = ["66percent", "70percent"]
        noise_only = noises == np.array(keepers.pop())
        for keep in keepers:
            noise_only = noise_only | (noises == np.array(keep))
        noise_only = np.arange(noises.shape[0])[noise_only]
            ## Convert bool mask to int for sparse array compatiblity

        noises = noises[noise_only]
        Xsp = Xsp[noise_only,:]

        print("After filteration.")
        print("\tNumber of TRs: {0}".format(trs.shape))
        print("\tNumber of noises: {0}".format(noises.shape))
        print("\tXsp shape: {0}".format(Xsp.shape))


        # ----
        # Smooth (and desparse)
        X = smooth(Xsp.todense(), tr=1.5, ub=0.06, lb=0.001)
        
        # ----
        # Shift labels and X by 3 TR (4.5 s) to account for 
        # HRF lag.
        lag = 0
        X, noises = _shiftby(X, noises, lag)
    
        print("After lag adjustment.")
        print("\tNumber of noises: {0}".format(noises.shape))
        print("\tX shape: {0}".format(X.shape))

        print("Label counts:")
        for resp in np.unique(noises):
            print("\t({0}, {1})".format(resp, np.sum(noises == resp)))

        # ----
        # Create train/class paritions
        nrow, ncol = X.shape
        cv = KFold(nrow, n_folds=5, indices=True)
        
        # ----
        # Loop over the folds, class as we go
        print("Classifying.")
        i = 0
        y = create_y(noises)
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
                acc_table_name, [roi, roiname, "cv_"+str(i)],
                accuracy)
            
            # ----
            # Update the Cv counter
            i += 1
    
    print("****")
    omean = np.array(overall_accuracy).mean()
    print("Overall accuracy for {1} was {0}".format(omean, roi))

    print("Saving accuracy")
    _save_accuracy_table(acc_table_name, [roi, roi, "overall"], omean)


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
            "right_hippocampus"]
    
    rois2 = ["parahippo_anterior",
            "parahippo_posterior"]

    [main(roi) for roi in rois2]

