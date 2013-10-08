import sys, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

from fmrilearn.analysis import eva
from fmrilearn.load import load_meta
from fmrilearn.load import load_nii
from fmrilearn.load import load_roifile
from fmrilearn.load import load_meta
from fmrilearn.save import save_tcdf
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.reshape import by_trial
from fmrilearn.preprocess.split import by_labels
from fmrilearn.preprocess.labels import construct_targets
from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets

from wheelerdata.load.fh import get_roi_data_paths
from wheelerdata.load.fh import get_metapaths_containing
from wheelerdata.load.fh import get_motor_metadata_paths

from fmrilearnexp.common import get_roiname
from fmrilearnexp.common import join_by_underscore


class DecomposeFH(object):
    """A facehouse decomposition experiment."""
    def __init__(self, spacetime, window=11, nsig=3):
        super(DecomposeFH, self).__init__()
        
        self.spacetime = spacetime
        self.window = window
        self.nsig = nsig

    def run(self, basename, roi, cond, smooth=False, filtfile=None):
        # Save here....
        table = join_by_underscore(False, basename, roi, cond)

        # and reinit write flags.
        roicount = 0
        mode = 'w'
        header = True

        # Getting to work, find subjects data
        paths = get_roi_data_paths(roi)
        metas = get_metapaths_containing(cond)

        # And decompose it
        for path, meta in zip(paths, metas):
            roiname = get_roiname(path)
            print("\t{0}".format(roiname))  ## ...progress

            # If were past the first Ss data, append.
            if roicount > 0:
                mode = 'a'
                header = False    

            # Get metadata
            trs, trial_index, y = load_meta(["TR", "trialcount", cond], meta)
            targets = construct_targets(trial_index=trial_index, trs=trs, y=y)

            # and data, preprocess too,
            X = load_nii(path, clean=True, sparse=False, smooth=smooth)
            X = X[trs,:]

            if filtfile is not None:
                filterX(filtfile, X, targets)

            # finally decompose.
            Xcs, csnames = self.spacetime.fit_transform(
                    X, targets["y"], targets["trial_index"], 
                    self.window, norm=True)
            
            # Name them,
            csnames = ["all", ] + sorted(np.unique(targets["y"]))
            
            # and write.
            for Xc, csname in zip(Xcs, csnames):
                save_tcdf(
                        name=join_by_underscore(True, table, csname), 
                        X=Xc, 
                        cond=csname,
                        dataname=join_by_underscore(False, roiname, 
                                self.spacetime.decompr.n_components),
                        index='auto',
                        header=header, 
                        mode=mode,
                        float_format="%.{0}f".format(self.nsig))
            roicount += 1


class Spacetime(object):
    """Decompose in spacetime. Requires a sklearn.decomposition instance."""
    
    def __init__(self, decompr):
        super(Spacetime, self).__init__()
        self._check_decompr(decompr)
        self.decompr = decompr


    def _check_decompr(self, decompr):
        if not hasattr(decompr, "fit"):
            raise AttributError("No fit method found on decompr")
        if not hasattr(decompr, "transform"):
            raise AttributError("No transform method found on decompr")


    def _ft(self, X):
        """fit_transform workhorse""" 
        
        Xc = self.decompr.fit_transform(X)

        checkX(Xc)
        if Xc.shape[0] != X.shape[0]:
            raise ValueError("After transform wrong row number.")
        if Xc.shape[1] > self.decompr.n_components:
            raise ValueError("Too many components")

        return Xc


    def fit_transform(self, X, y, trial_index, window, norm=True):
        """Converts X into Xtrial form (where the features are 
        individual trials (n_trials, window)) and decomposes 
        that matrix, possibly several times depending on y.
    
        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose
        y : 1D array, None by default
            Sample labels for the data
        trial_index : 1D array (n_sample, )
            Each unique entry should match a trial.
        window : int 
            Trial length
        norm : boolean, True by default
            Norm Xtrial feature level std dev

        Return
        ------
        Xcs : a list of 2D arrays (n_sample, n_components)
            The components for 'all' and (optionally) each unique y.
        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        csnames = []
        unique_y = sorted(np.unique(y))

        # Reshape by trials
        Xtrial, feature_names = by_trial(X, trial_index, window, y)
        if norm:
            scale(Xtrial.astype(np.float), axis=0, with_mean=False, copy=False)

        # and split by unique_y
        Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

        # put all that together
        Xtrials.append(Xtrial)
        Xtrials.extend([Xl.transpose() for Xl in Xlabels])

        # and decompose.
        Xcs = [self._ft(Xt) for Xt in Xtrials]

        # Create names
        csnames = ["all", ] + unique_y

        return Xcs, csnames


class Space(Spacetime):
    """Decompose in space. Requires a sklearn.decomposition instance."""
        
    def __init__(self, decompr):
        super(Space, self).__init__(decompr)


    def fit_transform(self, X, y, trial_index, window, mode="eva"):
        """Converts X into time-avearage trials and decomposes 
        that matrix, possibly several times depending on y.

        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose
        y : 1D array, None by default
            Sample labels for the data
        trial_index : 1D array (n_sample, )
            Each unique entry should match a trial.
        window : int 
            Trial length
        eva : str, 'eva' by default
            Select the trial averaging method

        Return
        ------
        Xcs : a list of 2D arrays (n_sample, n_components)
            The components for 'all' and (optionally) each unique y.
        csnames : 1D array
            The names of the components matrices
        """
        
        Xtrials = []
        csnames = []
        unique_y = sorted(np.unique(y))

        # Time averaged trials become features
        if mode == "eva":
            Xtrial, feature_names = eva(X, y, trial_index, window)
        elif mode == "fir":
            raise NotImplementedError("fir implementation needs work")
        else:
            raise ValueError("mode not understaood ('eva', 'fir')")

        # Split by unique_y,
        # put it all togther,
        Xtrials.append(Xtrial)
        for yi in unique_y:
            Xtrials.append(Xtrial[:, yi == feature_names])

        # and decompose
        Xcs = [self._ft(Xt) for Xt in Xtrials]

        # Create names
        csnames = ["all", ] + unique_y

        return Xcs, csnames        
        

class Time(Spacetime):
    """Decompose in time. Requires a sklearn.decomposition instance."""

    def __init__(self, decompr):
        super(Time, self).__init__(decompr)
        

    def fit_transform(self, X, y, trial_index, window, norm=True):
        """Converts X into Xtrial for each feature and decomposes 
        each matrix, possibly several times depending on y.

        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose
        y : 1D array, None by default
            Sample labels for the data
        trial_index : 1D array (n_sample, )
            Each unique entry should match a trial.
        window : int 
            Trial length
        norm : boolean, True by default
            Norm Xtrial feature level std dev

        Return
        ------
        Xcs : a list of 2D arrays (n_sample, n_components)
            The components for 'all' and (optionally) each unique y.
        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        csnames = []
        unique_y = sorted(np.unique(y))

        # Each feature by trials,
        for j in range(X.shape[1]):
            xj = X[:,j][:,np.newaxis]  ## Need 2D

            Xtrial, feature_names = by_trial(xj, trial_index, window, y)
            if norm:
                scale(Xtrial.astype(np.float), 
                        axis=0, with_mean=False, copy=False)
            
            # then by cond,
            Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)
            
            # put all that together,
            Xtrials.append(Xtrial)
            Xtrials.extend([Xl.transpose() for Xl in Xlabels])

            # names too.
            csnames.append(join_by_underscore(False, "all", j))
            csnames.extend([join_by_underscore(False, uy, j) for 
                    uy in unique_y])

        # And decompose after making 
        # sure there is enough data.
        Xcs = []
        bignames = []
        for Xt, csname in zip(Xtrials, csnames):
            if Xt.shape[1] > self.decompr.n_components:
                Xcs.append(self._ft(Xt))
                bignames.append(csname)

        return Xcs, bignames

