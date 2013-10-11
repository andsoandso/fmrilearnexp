import sys, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import scale

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
            
            # Try to count components for the dataname
            try:
                dataname = join_by_underscore(False, roiname, 
                        self.spacetime.estimator.n_components)
            except AttributeError:
                dataname = join_by_underscore(False, roiname)

            # and write.
            for Xc, csname in zip(Xcs, csnames):
                save_tcdf(
                        name=join_by_underscore(True, table, csname), 
                        X=Xc, 
                        cond=csname,
                        dataname=dataname,
                        index='auto',
                        header=header, 
                        mode=mode,
                        float_format="%.{0}f".format(self.nsig))
            roicount += 1


class Spacetime(object):
    """Decompose in spacetime.

    Parameters
    ----------
    estimator : a sklearn estimator object
        Must implement fit_transform (if in mode='decompose') or 
        fit_predict (if mode='cluster')

    mode : str ('decompose' by default)
        Decompose or cluster X?
    """
    
    def __init__(self, estimator, mode='decompose'):
        super(Spacetime, self).__init__()

        self.mode = mode
        self.estimator = estimator


    def _fp(self, X):
        """The cluster workhorse

        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose

        Return
        ------
        Xc - 2D array-like (n_sample, n_clusters)
        """

        nrow = X.shape[0]

        clabels = self.estimator.fit_predict(X.transpose())
        uclabels = sorted(np.unique(clabels))
        
        # Average cluster examples, making Xc
        Xc = np.zeros((nrow, len(unique_cl)))         ## Init w/ 0
        for i, ucl in enumerate(uclabels):
            Xtc[:,i] = X[:,ucl == uclabels].mean(1)  ## Select and avg

        assert checkX(Xc)
        assert Xc.shape[0] == X.shape[0], ("After transform wrong row number")
        assert Xc.shape[1] == len(unique_cl), ("Afer transform" 
            " wrong col number")

        return Xc


    def _ft(self, X):
        """The decompose workhorse

        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose

        Return
        ------
        Xc - 2D array-like (n_sample, n_components)
        """ 
        
        Xc = self.estimator.fit_transform(X)

        assert checkX(Xc)
        assert Xc.shape[0] == X.shape[0], ("After transform wrong row number")

        # The n_components attr is optional
        try: 
            assert Xc.shape[1] <= self.estimator.n_components, ("Too many" 
                "components")
        except AttributeError:
            pass

        return Xc


    def fit_transform(self, X, y, trial_index, window, norm=True):
        """Converts X into Xtrial form (where the features are  individual
        trials (n_trials, window)) and decomposes  that matrix, possibly
        several times depending on y.
    
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
        if self.mode == 'decomp':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]

        # Create names
        csnames = ["all", ] + unique_y

        return Xcs, csnames


class Space(Spacetime):
    """Decompose in space.

    estimator : a sklearn estimator object
        Must implement fit_transform (if in mode='decompose') or 
        fit_predict (if mode='cluster')

    avgfn : an averaging fn (see Note)

    mode : str ('decompose' by default)
        Decompose or cluster X?

    Note
    ----
    Requires an average function with a signature like: 

        Xtrial, feature_names = avg(X, y, trial_index, window, norm=True)

    Where the Xtrial matrix is shaped as (window, n_trial * n_unique_y). 
    See  `fmrilearn.analysis.eva` for an example.
    """
        
    def __init__(self, estimator, avgfn, mode="decompose"):
        super(Space, self).__init__(estimator, mode)

        self.avgfn = avgfn


    def fit_transform(self, X, y, trial_index, window, norm=True):
        """Converts X into time-avearage trials and decomposes  that
        matrix, possibly several times depending on y.

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

        norm : False
            A dummy argument

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

        Xtrial, feature_names = self.avgfn(
                X, y, trial_index, window, norm=True)

        # Split by unique_y, put it all togther,
        Xtrials.append(Xtrial)
        for yi in unique_y:
            Xtrials.append(Xtrial[:, yi == feature_names])

        # and decompose
        Xcs = [self._ft(Xt) for Xt in Xtrials]

        # Create names
        csnames = ["all", ] + unique_y

        return Xcs, csnames        
        

class AverageTime(object):
    """Average trials. 

    Parameters
    ----------
    Requires an average function with a signature like: 

    Xtrial, feature_names = avg(X, y, trial_index, window, norm=True)

    Where the Xtrial matrix is shaped as (window, n_trial * n_unique_y).
    See  `fmrilearn.analysis.eva` for an example.
    """

    def __init__(self, avgfn):
        super(AverageTime, self).__init__()

        self.avgfn = avgfn


    def fit_transform(self, X, y, trial_index, window, norm=True):
        """Average X by trial based on y (and trial_index).

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
        Xavgs : a list of 2D arrays (n_sample, n_components)
            The averaged trials for 'all' and (optionally) each unique y.

        avgames : 1D array
            The names of the components matrices"""

        Xavgs = []
        avgnames = []
        unique_y = sorted(np.unique(y))

        # Time averaged trials become features
        Xavg, feature_names = self.avgfn(X, y, trial_index, window, norm=True)

        # Split by unique_y, put it all togther,
        Xavgs.append(Xavg)
        for yi in unique_y:
            Xavgs.append(Xavg[:, yi == feature_names])

        avgnames = ["all", ] + unique_y

        return Xavgs, avgnames


class Time(Spacetime):
    """Decompose in time.

    Parameters
    ----------
    estimator : a sklearn estimator object
        Must implement fit_transform (if in mode='decompose') or 
        fit_predict (if mode='cluster')

    mode : str ('decompose' by default)
        Decompose or cluster X?
    """

    def __init__(self, estimator, mode="decompose"):
        super(Time, self).__init__(estimator, mode)
        

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
            # If no n_components always pass
            try: 
                nc = self.estimator.n_components
            except AttributeError:
                nc = X.shape[1] - 1  

            if Xt.shape[1] > nc:
                Xcs.append(self._ft(Xt))
                bignames.append(csname)

        return Xcs, bignames

