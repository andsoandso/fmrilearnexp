import sys, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from fmrilearn.load import load_meta
from fmrilearn.load import load_nii
from fmrilearn.load import load_roifile
from fmrilearn.save import save_tcdf
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.data import filterX
from fmrilearn.preprocess.data import smooth as smoothfn
from fmrilearn.preprocess.reshape import by_trial
from fmrilearn.preprocess.split import by_labels
from fmrilearn.preprocess.labels import unique_sorted_with_nan
from fmrilearn.preprocess.labels import construct_targets
from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets
from fmrilearn.preprocess.labels import csv_to_targets
from fmrilearn.preprocess.labels import targets_to_csv
from fmrilearn.preprocess.labels import tr_pad_targets
from fmrilearn.preprocess.labels import reprocess_targets

    
from wheelerdata.load.fh import FH
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
                X, targets = filterX(filtfile, X, targets)
            if smooth:
                X = smoothfn(X, tr=1.5, ub=0.10, lb=0.001)

            # finally decompose.
            Xcs, csnames = self.spacetime.fit_transform(
                    X, targets["y"], targets["trial_index"], 
                    self.window)
            
            # Name them,
            csnames = sorted(np.unique(targets["y"]))
            csnames = unique_sorted_with_nan(csnames)

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


class DecomposeExp(object):
    """A decomposition experiment."""
    def __init__(self, spacetime, data, window=11, nsig=3, tr=1.5):
        super(DecomposeExp, self).__init__()
        
        self.spacetime = spacetime
        self.window = window
        self.nsig = nsig
        self.tr = tr
        self.data = data


    def run(self, basename, roi, cond, smooth=False, filtfile=None, event=False):
        # Save here....
        table = join_by_underscore(False, basename, roi, cond)

        # and reinit write flags.
        roicount = 0
        mode = 'w'
        header = True

        # Getting to work, find subjects data
        paths = self.data.get_roi_data_paths(roi)
        if not event:
            metas = self.data.get_metapaths_containing(cond)
        else:
            metas = self.data.get_RT_metadata_event_paths()

        # And decompose it
        for path, meta in zip(paths, metas):
            roiname = get_roiname(path)
            print("\t{0}".format(roiname))  ## ...progress

            # If were past the first Ss data, append.
            if roicount > 0:
                mode = 'a'
                header = False

            # Get all metadata
            targets = csv_to_targets(meta)

            # and data, preprocess too,
            X = load_nii(path, clean=True, sparse=False, smooth=smooth)
            targets = tr_pad_targets(targets, "TR", X.shape[0], pad=np.nan)

            targets_to_csv(targets, "{0}_targets_before.csv".format(basename))
            if filtfile is not None:
                targets = reprocess_targets(filtfile, targets, np.nan)
                assert targets["TR"].shape[0] == X.shape[0], "target reprocessing is broken"
            targets_to_csv(targets, "{0}_targets_after.csv".format(basename))

            Xcs, csnames = self.spacetime.fit_transform(
                    X, targets[cond], targets["trialcount"], 
                    self.window, self.tr)

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


class Decompose(object):
    """A template for decomposition objects.

    The only public method should be `fit_transform` which
    has the a signture like:

        fit_transform(self, X, y, trial_index, window)
    """

    def __init__(self, estimator, mode='decompose'):
        super(Decompose, self).__init__()

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
        uclabels = unique_sorted_with_nan(uclabels)

        # Average cluster examples, filling Xc
        Xc = np.zeros((nrow, len(uclabels)))         ## Init w/ 0
        for i, ucl in enumerate(uclabels):
            Xc[:,i] = X[:,ucl == clabels].mean(1)

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
        raise NotImplementedError("Subclass `Decompose` then implement this")


class Spacetime(Decompose):
    """Decompose trials in spacetime.

    Parameters
    ----------
    estimator : a sklearn estimator object
        Must implement fit_transform (if in mode='decompose') or 
        fit_predict (if mode='cluster')

    mode : str ('decompose' by default)
        Decompose or cluster X?
    """
    

    def __init__(self, estimator, mode='decompose'):
        super(Spacetime, self).__init__(estimator, mode)


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
            The components for each unique y.

        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        Xcs = []
        csnames = []

        # Reshape by trials, rescale too
        Xtrial, feature_names = by_trial(X, trial_index, window, y)
        unique_fn = sorted(np.unique(feature_names))
        unique_fn = unique_sorted_with_nan(unique_fn)

        scaler = MinMaxScaler(feature_range=(0, 1))
        Xtrial = scaler.fit_transform(Xtrial.astype(np.float))

        # and split by unique_y
        Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

        # put all that together
        Xtrials.extend([Xl.transpose() for Xl in Xlabels])

        # and decompose.
        if self.mode == 'decompose':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]
        else:
            raise ValueError("mode not understood.")

        return Xcs, 


class Voxel(Decompose):
    """Decompose voxels, then break into trials.
    
    Completely overides Spacetime.run()

    Parameters
    ----------
    estimator : a sklearn estimator object
        Must implement fit_transform (if in mode='decompose') or 
        fit_predict (if mode='cluster')

    mode : str ('decompose' by default)
        Decompose or cluster X?
    """

    def __init__(self, estimator, mode="decompose"):
        super(Voxel, self).__init__(estimator, mode)


    def fit_transform(self, X, y, trial_index, window):
        Xcs = []
        csnames = []

        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X.astype(np.float))
        
        if self.mode == 'decompose':
            Xc = self._ft(X)
        elif self.mode == 'cluster':
            Xc = self._fp(X)
        else:
            raise ValueError("mode not understood.")

        # Create Xcs
        csnames = sorted(np.unique(y))
        csnames = unique_sorted_with_nan(csnames)

        for j in range(len(csnames)):
            Xcs.append(np.zeros([window, Xc.shape[1]]))

        for j in range(Xc.shape[1]):
            xc = Xc[:,j]
            xc = xc[:,np.newaxis]
            Xtrial, feature_names = by_trial(xc, trial_index, window, y)
            unique_fn = sorted(np.unique(feature_names))
            unique_fn = unique_sorted_with_nan(unique_fn)

            # For the current comp j,
            # split up into Xtrials and
            # average each
            Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)

            Xcs[0][:,j] = Xtrial.mean(1)    
            for i, xl in enumerate(Xlabels):
                Xcs[i+1][:,j] = xl.transpose().mean(1)
        
        return Xcs, csnames


class Space(Decompose):
    """Decompose trials in space.

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


    def fit_transform(self, X, y, trial_index, window, tr):
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
            The components for each unique y.

        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        Xcs = []
        csnames = []
        
        Xtrial, feature_names = self.avgfn(X, y, trial_index, window, tr)
        unique_fn = sorted(np.unique(feature_names))
        unique_fn = unique_sorted_with_nan(unique_fn)

        # Loop over unique_y not unique_fn as we want to
        # pull apart what was in the orginal y, 
        # not anything that unique_fn may contain.
        # Xtrials.append(Xtrial)
        for yi in unique_fn:
            Xtrials.append(Xtrial[:, np.str(yi) == feature_names])

        # and decompose.
        if self.mode == 'decompose':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]
        else:
            raise ValueError("mode not understood.")

        return Xcs, unique_fn      
        

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


    def fit_transform(self, X, y, trial_index, window, tr):
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
            The averaged trials for each unique y.

        avgames : 1D array
            The names of the components matrices"""

        Xavgs = []
        avgnames = []

        # Time averaged trials become features
        Xavg, feature_names = self.avgfn(X, y, trial_index, window, tr)
        unique_fn = sorted(np.unique(feature_names))
        unique_fn = unique_sorted_with_nan(unique_fn)
        
        # Split by unique_y, put it all togther,
        Xavgsfull = []
        for yi in unique_fn:
            Xavgsfull.append(Xavg[:, np.str(yi) == feature_names]) ## DEBUG, save all FIRs
            Xavgs.append(
                Xavg[:, np.str(yi) == feature_names].mean(1)[:,np.newaxis])

        # DEBUG - save all FIRs
        for yi, Xa in zip(unique_fn, Xavgsfull):
            np.savetxt("firs-{0}.txt".format(yi), Xa, fmt="%1.8f")

        return Xavgs, unique_fn


class Time(Decompose):
    """Decompose trial in time.

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
            The components for each unique y.

        csnames : 1D array
            The names of the components matrices
        """

        Xtrials = []
        csnames = []
        unique_y = sorted(np.unique(y))
        unique_y = unique_sorted_with_nan(unique_y)

        # Each feature by trials,
        for j in range(X.shape[1]):
            xj = X[:,j][:,np.newaxis]  ## Need 2D

            Xtrial, feature_names = by_trial(xj, trial_index, window, y)
            scaler = MinMaxScaler(feature_range=(0, 1))
            Xtrial = scaler.fit_transform(Xtrial.astype(np.float))
            
            # then by cond,
            Xlabels, _ = by_labels(X=Xtrial.transpose(), y=feature_names)
            
            # put all that together,
            Xtrials.extend([Xl.transpose() for Xl in Xlabels])

            # names too.
            csnames.extend([join_by_underscore(False, uy, j) for 
                    uy in unique_y])

        # And decompose after making 
        # sure there is enough self.data.
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

