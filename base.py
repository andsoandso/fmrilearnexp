import sys, os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from simfMRI.noise import white

from fmrilearn.load import load_meta
from fmrilearn.load import load_nii
from fmrilearn.load import load_roifile
from fmrilearn.save import save_tcdf
from fmrilearn.preprocess.data import checkX
from fmrilearn.preprocess.data import filterX
from fmrilearn.preprocess.data import smooth as smoothfn
from fmrilearn.preprocess.reshape import by_trial
from fmrilearn.preprocess.split import by_labels
# from fmrilearn.preprocess.labels import unique_sorted_with_nan
from fmrilearn.preprocess.labels import unique_nan
from fmrilearn.preprocess.labels import sort_nanfirst
from fmrilearn.preprocess.labels import construct_targets
from fmrilearn.preprocess.labels import construct_filter
from fmrilearn.preprocess.labels import filter_targets
from fmrilearn.preprocess.labels import csv_to_targets
from fmrilearn.preprocess.labels import targets_to_csv
from fmrilearn.preprocess.labels import tr_pad_targets
from fmrilearn.preprocess.labels import reprocess_targets

from wheelerdata.load.fh import FH
from wheelerdata.load.simulated import make_bold
from wheelerdata.load.reproduced import make_bold as make_bold_re

from fmrilearnexp.common import get_roiname
from fmrilearnexp.common import join_by_underscore


class DecomposeExpReproduction(object):
    def __init__(self, spacetime, data, window=11, nsig=3, tr=1.5):
        super(DecomposeExpReproduction, self).__init__()

        self.spacetime = spacetime
        self.data = data
        self.window = window
        self.nsig = nsig
        self.tr = tr
        
    def run(self, basename, cond, index, wheelerdata, cond_to_rt, 
        smooth=False,
        filtfile=None, TR=2, trname="TR", 
        n_features=10, n_univariate=None, n_accumulator=None, n_decision=None, 
        n_noise=None, drift_noise=False, step_noise=False, z_noise=False,
        drift_noise_param=None, step_noise_param=None, z_noise_param=None,
        noise_f=white, hrf_f=None, hrf_params=None, prng=None):       
        """Reproduce the cond from the wheelerdata experiment
        
        Parameters
        ---------
        basename : str
            The name for the Reproduced datafile, will be suffixed
            by each cond and scode and .csv 
            (i.e. `'{0}_{1}_{2}.csv'.format(basename, cond, scode)`).
        cond : str
            A condition name found in the wheelerdata objects metadata
        index : str
            A name of a trial index found in the wheelerdata object metadata
        wheelerdata : object, instance of Wheelerdata
            A Wheelerdata object
        cond_to_rt: dict
            A map of cond (key) to reaction time (item, (int, float))    
        smooth : boolean, optional
            Do bandpass filtering (default False)
        filtfile : str, None
            A name of json file designed for reprocessing Wheelerdata metadata
        TR : float, int
            The repitition time of the experiement
        trname : str
            The name of the index of TRs in the metadata
        n_features : int
            The number of features in total (other n_* arguements
            must sum to this value
        n_univariate : int
            The number of univariate (boxcar) features
        n_accumulator : int
            The number of accumulator features
        n_decision : int
            The number of decision features
        n_noise : int
            The number of noise features
        drift_noise : boolean, optional
            Add noise to the drift rate of the accumulator features
        step_noise : boolean, optional
            Add Noise to each step accumulator features
        z_noise : boolean, optional
            Add noise to the start value of accumulator features
        drift_noise_param : None or dict, optional
            Parameters for drift_noise which is drawn from a
            Gaussian distribution. None defaults to: 
            `{"loc": 0, "scale" : 0.5}`
        step_noise_param : None or dict, optional
            Parameters for step_noise which is drawn from a 
            Gaussian distribution. None defaults to:
            `{"loc" : 0, "scale" : 0.2, "size" : 1}`
        z_noise_param : None or dict, optional
            Parameters for z_noise which is drawn from the uniform
            distribution. None defaults to:
            `{"low" : 0.01, "high" : 0.5, "size" : 1}`
        noise_f : function, optional
            Produces noise, must have signatures like `noise, prng = f(N, prng)`
        hrf_f : function, optional
            Returns a haemodynamic response, signature hrf_f(**hrf_params)
        hrf_params : dict
            Keyword parameters for hrf_f
        prng : None or RandomState object
            Allows for independent random draws, used for all 
            random sampling
        """

        mode = 'w'
        header = True

        # All *s lists correspond to wheelerdata.scodes
        scodes = self.data.scodes
        Xs, ys, yindices = make_bold_re(
                cond, index, self.data,
                cond_to_rt,
                filtfile=filtfile, 
                trname=trname,
                noise_f=noise_f, 
                hrf_f=hrf_f, 
                hrf_params=hrf_params, 
                n_features=n_features, 
                n_univariate=n_univariate, 
                n_accumulator=n_accumulator, 
                n_decision=n_decision, 
                n_noise=n_noise, 
                drift_noise=drift_noise, 
                step_noise=step_noise, 
                z_noise=z_noise,
                drift_noise_param=drift_noise_param, 
                step_noise_param=step_noise_param, 
                z_noise_param=z_noise_param,
                prng=prng)
        
        for scode, X, y, yindex in zip(scodes, Xs, ys, yindices):
            if smooth:
                X = smoothfn(X, tr=1.5, ub=0.10, lb=0.001)
            
            # Normalize
            norm = MinMaxScaler((0,1))
            X = norm.fit_transform(X.astype(np.float))
            
            Xcs, csnames = self.spacetime.fit_transform(
                    X, y, yindex, self.window, self.tr)

            # Name them,
            csnames = unique_nan(y)
            csnames = sort_nanfirst(csnames)

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
            
            # After s 1 go to append mode
            mode = 'a'
            header = False


class DecomposeSimulation(object):
    """A Monte Carlo simulation of the facehouse decomposition experiment"""
    def __init__(self, spacetime, window=11, nsig=3):
        super(DecomposeSimulation, self).__init__()
        
        self.spacetime = spacetime
        self.window = window
        self.nsig = nsig


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
            
            # Normalize
            norm = MinMaxScaler((0,1))
            X = norm.fit_transform(X.astype(np.float))
            
            # finally decompose.
            Xcs, csnames = self.spacetime.fit_transform(
                    X, targets["y"], targets["trial_index"], 
                    self.window)
            
            # Name them,
            csnames = unique_nan(y)
            csnames = sort_nanfirst(csnames)

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

            #targets_to_csv(targets, "{0}_targets_before.csv".format(basename))
            if filtfile is not None:
                targets = reprocess_targets(filtfile, targets, np.nan)
                assert targets["TR"].shape[0] == X.shape[0], ("target" 
                    "reprocessing is broken")
            #targets_to_csv(targets, "{0}_targets_after.csv".format(basename))
            
            # Normalize
            norm = MinMaxScaler((0,1))
            X = norm.fit_transform(X.astype(np.float))
            
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
            known = []
            for Xc, csname in zip(Xcs, csnames):
                if not csname in known:
                    save_tcdf(
                            name=join_by_underscore(True, table, csname), 
                            X=Xc, 
                            cond=csname,
                            dataname=dataname,
                            index='auto',
                            header=header, 
                            mode=mode,
                            float_format="%.{0}f".format(self.nsig))
                    known.append(csname)
                else:
                    save_tcdf(
                            name=join_by_underscore(True, table, csname), 
                            X=Xc, 
                            cond=csname,
                            dataname=dataname,
                            index='auto',
                            header=False, 
                            mode='a',
                            float_format="%.{0}f".format(self.nsig))
            roicount += 1
        
        
class Decompose(object):
    """A template for decomposition objects.

    The only public method should be `fit_transform` which
    has the a signature like:

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
        uclabels = unique_nan(clabels)
        uclabels = sort_nanfirst(uclabels)
        # uclabels = sorted(np.unique(clabels))
        # uclabels = unique_sorted_with_nan(uclabels)

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
        raise NotImplementedError("Subclass `Decompose` and implement")


class Timecourse(object):
    """Decompose full voxel timecourses."""
     
    def __init__(self, estimator, mode):        
        self.mode = mode
        self.estimator = estimator
        
    def fit_transform(self, X, y, trial_index, window, tr):
        if self.mode == 'decompose': 
            Xc = self.estimator.fit_transform(X)
        elif self.mode == 'cluster':
            # Use cluster labels to create average timecourses
            clabels = self.estimator.fit_predict(X.transpose())
            uclabels = unique_nan(clabels)
            uclabels = sort_nanfirst(uclabels)         

            Xc = np.zeros((X.shape[0], len(uclabels))) ## Init
            for i, ucl in enumerate(uclabels):
                Xc[:,i] = X[:,ucl == clabels].mean(1)
        else:
            raise ValueError("mode not understood.")

        unique_y = sort_nanfirst(unique_nan(y))        
        Xcs = [Xc[uy == unique_y,:] for uy in unique_y]        
        
        return Xcs, unique_y


class Trialtime(Decompose):
    """Decompose each trial across voxels."""
    
    def __init__(self, estimator, mode='decompose'):
        super(Trialtime, self).__init__(estimator, mode)        
    
    def fit_transform(self, X, y, trial_index, window, tr):
        Xcs = []
        ycs = []
        uti = unique_nan(trial_index)
        uti = uti[np.logical_not(np.isnan(uti))]
        
        for n, ti in enumerate(uti):
            # Skip last trial to prevent padding overflow
            if n+1 == len(uti):
                break
            
            # Locate trial and either 
            # extend l to window, if needed
            # or shorten each trial to window, if needed
            mask = trial_index == ti
            l = np.sum(mask)
            if l < window:
                i = 1
                for j, ma in enumerate(mask):
                    if ma:
                        i += 1
                    if i > l:
                        pad = window - l + 1
                        mask[j:(j + pad)] = True
                        break
            elif window < l:
                i = 0
                for j, ma in enumerate(mask):
                    if ma:
                        i += 1
                    if i > window:
                       mask[j] = False 
                        
            Xtrial = X[mask,:]
            from simfMRI.norm import zscore
            Xtrial = zscore(Xtrial)
            
            assert Xtrial.shape == (window, X.shape[1]), "Xtrial wrong shape"
            
            if self.mode == 'decompose':
                Xcs.append(self.estimator.fit_transform(Xtrial))
            elif self.mode == 'cluster':
                # Useluster labels to create average timecourses
                clabels = self.estimator.fit_predict(Xtrial.transpose())
                uclabels = unique_nan(clabels)
                uclabels = sort_nanfirst(uclabels)
                
                Xc = np.zeros((Xtrial.shape[0], len(uclabels))) ## Init w/ 0
                for i, ucl in enumerate(uclabels):
                    Xc[:,i] = Xtrial[:,ucl == clabels].mean(1)
                Xcs.append(Xc)
            
            ycs.append(y[trial_index == ti][0])
        
        assert len(Xcs) == len(ycs), ("Xcs and ycs mismatch")
        
        return Xcs, np.asarray(ycs) 
        

class Space(Decompose):
    """Calculate average trials by cond for each voxel, and reduce
    across voxels."""
        
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
        unique_fn = sort_nanfirst(unique_nan(feature_names))

        # Split up by feature_names
        for yi in unique_fn:
            Xtrials.append(Xtrial[:, feature_names == yi])
                
        # and decompose.
        if self.mode == 'decompose':
            Xcs = [self._ft(Xt) for Xt in Xtrials]
        elif self.mode == 'cluster':
            Xcs = [self._fp(Xt) for Xt in Xtrials]
        else:
            raise ValueError("mode not understood.")

        return Xcs, unique_fn
        

class AverageTimecourse(Decompose):
    """Average X timecourse and decompose that."""

    def __init__(self, estimator, mode="decompose"):
        super(AverageTimecourse, self).__init__(estimator, mode)

        self.avgfn = avgfn


    def fit_transform(self, X, y, trial_index, window, tr):
        """Average X by trial based on y (and trial_index).

        Parameters
        ----------
        X : 2D array-like (n_sample, n_feature)
            The data to decompose

        y : 1D array, None by default
            Sample labels for the data

        trial_index : Dummy

        window : Dumy
        
        tr: Dumy

        Return
        ------
        Xcs : TODO

        ycs : TODO
        """

        Xa = X.mean(1)[:,np.newaxis]
        
        if self.mode == 'decompose':
            Xc = self._ft(Xa)
        elif self.mode == 'cluster':
            Xc = self._fp(Xa)
        else:
            raise ValueError("mode not understood.")
        
        unique_y = sort_nanfirst(unique_nan(y))        
        Xcs = [Xc[uy == unique_y,:] for uy in unique_y]

        return Xcs, unique_y
