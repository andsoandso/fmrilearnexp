import sys, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from fmrilearn.load import load_meta
from fmrilearn.load import load_nii
from fmrilearn.load import load_roifile
from fmrilearn.load import load_meta
from fmrilearn.save import save_tcdf
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.data import smooth as smoothfn
from fmrilearn.preprocess.reshape import by_trial
from fmrilearn.preprocess.split import by_labels
from fmrilearn.preprocess.labels import construct_targets
from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets

from wheelerdata.load.fh import get_roi_data_paths
from wheelerdata.load.fh import get_metapaths_containing
from wheelerdata.load.fh import get_motor_metadata_paths
from wheelerdata.load.simulated import make_bold

from fmrilearnexp.common import get_roiname
from fmrilearnexp.common import join_by_underscore


class DecomposeSimulation(object):
    """A simulation of the facehouse decomposition experiment"""
    def __init__(self, spacetime, window=11, nsig=3):
        super(DecomposeSimulation, self).__init__()
        
        self.spacetime = spacetime
        self.window = window
        self.nsig = nsig

        # Create a random seed for repproducublity.
        # Need to mod make_bold to accept it

    def run(self, basename, smooth=False, filtfile=None, 
        n=None, tr=None, n_rt=None, n_trials_per_cond=None,
        durations=None ,noise=None, n_features=None, n_univariate=None, 
        n_accumulator=None, n_decision=None, n_noise=None, 
        n_repeated=None, drift_noise=False, step_noise=False):
        
        # Write init
        mode = 'w'
        header = True

        for scode in range(n):
            # If were past the first Ss data, append.
            if scode > 0:
                mode = 'a'
                header = False

            # Create the data
            X, y, y_trialcount = make_bold(
                    n_rt, 
                    n_trials_per_cond, 
                    tr, 
                    durations=durations, 
                    noise=noise, 
                    n_features=n_features, 
                    n_univariate=n_univariate, 
                    n_accumulator=n_accumulator, 
                    n_decision=n_decision,
                    n_noise=n_noise,
                    n_repeated=n_repeated,
                    drift_noise=drift_noise,
                    step_noise=step_noise)

            targets = construct_targets(trial_index=y_trialcount, y=y)

            # Drop baseline trials created by make_bold
            baselinemask = np.arange(y.shape[0])[y != 0]
            X = X[baselinemask, ]
            targets = filter_targets(baselinemask, targets)

            # Filter and
            if filtfile is not None:
                filterX(filtfile, X, targets)
            if smooth:
                X = smoothfn(X, tr=1.5, ub=0.10, lb=0.001)

            # finally decompose.
            Xcs, csnames = self.spacetime.fit_transform(
                    X, targets["y"], targets["trial_index"], 
                    self.window)
            
            # Name them,
            csnames = ["all", ] + sorted(np.unique(targets["y"]))

            # and write.
            for Xc, csname in zip(Xcs, csnames):
                save_tcdf(
                        name=join_by_underscore(True, basename, csname), 
                        X=Xc, 
                        cond=csname,
                        dataname=join_by_underscore(False, 
                                os.path.split(basename)[-1], scode),
                        index='auto',
                        header=header, 
                        mode=mode,
                        float_format="%.{0}f".format(self.nsig))


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
                    self.window)
            
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
        Xc = np.zeros((nrow, len(uclabels)))         ## Init w/ 0
        for i, ucl in enumerate(uclabels):
            clustermean = X[:,ucl == uclabels].mean(1)
            Xc[:,i] = clustermean  ## Select and avg

        assert checkX(Xc)
        assert Xc.shape[0] == X.shape[0], ("After transform wrong row number")
        assert Xc.shape[1] == len(uclabels), ("Afer transform" 
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


    def fit_transform(self, X, y, trial_index, window):
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

        Return
        ------
        Xcs : a list of 2D arrays (n_sample, n_components)
            The components for 'all' and (optionally) each unique y.

        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        Xcs = []
        csnames = []
        unique_y = sorted(np.unique(y))

        # Reshape by trials, rescale too
        Xtrial, feature_names = by_trial(X, trial_index, window, y)
        scaler = MinMaxScaler(feature_range=(0, 1))
        Xtrial = scaler.fit_transform(Xtrial.astype(np.float))

        # and split by unique_y
        Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

        # put all that together
        Xtrials.append(Xtrial)
        Xtrials.extend([Xl.transpose() for Xl in Xlabels])

        # and decompose.
        if self.mode == 'decompose':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]
        else:
            raise ValueError("mode not understood.")

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


    def fit_transform(self, X, y, trial_index, window):
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

        norm : True
            A dummy argument

        Return
        ------
        Xcs : a list of 2D arrays (n_sample, n_components)
            The components for 'all' and (optionally) each unique y.

        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        Xcs = []
        csnames = []
        # unique_y = sorted(np.unique(y))

        Xtrial, feature_names = self.avgfn(X, y, trial_index, window)
        unique_fn = sorted(np.unique(feature_names))

        # Split by unique_y, put it all togther,
        Xtrials.append(Xtrial)
        for yi in unique_fn:
            Xtrials.append(Xtrial[:, yi == feature_names])

        # and decompose.
        if self.mode == 'decompose':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]
        else:
            raise ValueError("mode not understood.")

        # Create names        
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


    def fit_transform(self, X, y, trial_index, window):
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
        Xavgs : a list of 2D arrays (n_sample, 1)
            The averaged trials for 'all' and (optionally) each unique y.

        avgames : 1D array
            The names of the components matrices"""

        Xavgs = []
        avgnames = []
        unique_y = sorted(np.unique(y))

        # Time averaged trials become features
        Xavg, feature_names = self.avgfn(X, y, trial_index, window)

        # Split by unique_y, put it all togther,
        Xavgs.append(Xavg.mean(1)[:,np.newaxis])
        for yi in unique_y:
            Xavgs.append(Xavg[:, yi == feature_names].mean(1)[:,np.newaxis])

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
        

    def fit_transform(self, X, y, trial_index, window):
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
            scaler = MinMaxScaler(feature_range=(0, 1))
            Xtrial = scaler.fit_transform(Xtrial.astype(np.float))
            
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
                # and decompose.
                if self.mode == 'decompose':
                    Xcs.append(self._ft(Xt))
                elif self.mode == 'cluster':
                    Xcs.append(self._fp(Xt))
                else:
                    raise ValueError("mode not understood.")

                bignames.append(csname)

        return Xcs, bignames

