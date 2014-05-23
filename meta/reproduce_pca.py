"""Reproduce by simulation of PCA estimation using wheelerdata"""

import sys, os
import numpy as np
from numpy.random import RandomState
import argparse
import json
from sklearn.decomposition import PCA

from simfMRI.noise import white

from wheelerdata.load.meta import get_data

from fmrilearn.analysis import fir
from fmrilearnexp.base import Space
from fmrilearnexp.base import DecomposeExpReproduction



# ---------------------------------------------------------------------------
# Process argv 
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
        description="Calculate average FIRs using simulated Wheelerdata")
parser.add_argument("--name", 
        help="The basename of this experiment")
parser.add_argument("--cond", 
        help=("Name from data to use as labels"))
parser.add_argument("--index",
        help=("Name from data for a trial index"))
parser.add_argument("--data", 
        help=("The name of the Wheelerdata set"))
parser.add_argument("--TR", type=float,
        help=("The TR of the data"))

parser.add_argument("--trname", default="TR", 
        help=("Name of the TRs in data"))
parser.add_argument("--smooth", default=False,
        help=("Smooth data before averaging"))
parser.add_argument("--filtfile", default=None, 
        help=("Filter the labels using based on a json file"))

parser.add_argument("--n_features", default=10, type=int,
        help=("The total number of features"))
parser.add_argument("--n_univariate", default=None, type=int,
        help=("The number of boxcar features"))
parser.add_argument("--n_accumulator", default=None, type=int,
        help=("The number of accumulator features"))
parser.add_argument("--n_decision", default=None, type=int,
        help=("The number of decision features"))
parser.add_argument("--n_noise", default=None, type=int,
        help=("The number of noise features"))
parser.add_argument("--drift_noise", default=False, type=bool,
        help=("Add noise to accumulator drift rate"))
parser.add_argument("--step_noise", default=False, type=bool,
        help=("Add noise to accumulator drift rate"))
parser.add_argument("--z_noise", default=False, type=bool,
        help=("Add noise to accumulator drift rate"))

parser.add_argument("--drift_noise_param", 
        default='{"loc": 0, "scale" : 0.5}', type=json.loads,
        help=("Modify Gaussian drift noise"))
parser.add_argument("--step_noise_param", 
        default='{"loc" : 0, "scale" : 0.2, "size" : 1}', type=json.loads,
        help=("Modify Gaussian step noise"))
parser.add_argument("--z_noise_param", 
        default='{"low" : 0.01, "high" : 0.5, "size" : 1}', type=json.loads,
        help=("Modify uniform start value noise"))

parser.add_argument("--seed", default=None,
        help=("Random seed value (initalizes a RandomState() instance"))
args = parser.parse_args()

prng = None
if args.seed != None:
    prng = RandomState(int(args.seed))

# RTs for Wheelerdata
def lookup_cond_to_dt(data):
    lookup = {
        "fh" : {
            "fast" : 2, "slow" : 4, "nan": 0, 
            1 : 2, 2 : 4, 0 : 0
            },
        "butterfly" : {
            "fast" : 4, "slow" : 6, "nan": 0,
            1 : 4, 2 : 6, 0 : 0
            },
        "clock" : {
            "fast" : 4, "slow" : 6, "nan": 0,
            1 : 4, 2 : 6, 0 : 0    
            },
        "polygon" : {
            "fast" : 1, "slow" : 2, "nan": 0,
            1 : 1, 2 : 2, 0 : 0
            },
        "redgreen" : {
            "fast" : 1, "slow" : 3, "nan": 0,
            1 : 1, 2 : 3, 0 : 0
            },
        "biasbox" : {
            "fast" : 1, "slow" : 2, "nan": 0,
            1 : 1, 2 : 2, 0 : 0
            }
    }
    return lookup[data]


# Lookup cond_to_rt
cond_to_rt = lookup_cond_to_dt(args.data)


# ---------------------------------------------------------------------------
# BOLD creation vars
# ---------------------------------------------------------------------------
noise_f=white
hrf_f=None
hrf_params=None

# ---------------------------------------------------------------------------
# Setup exp and run
# ---------------------------------------------------------------------------
data = get_data(args.data)

spacetime = Space(PCA(5, whiten=True), fir, mode="decompose")
exp = DecomposeExpReproduction(spacetime, data, 
        window=15, nsig=3, tr=args.TR)

exp.run(args.name, args.cond, args.index, data, cond_to_rt,
        smooth=args.smooth,
        filtfile=args.filtfile, 
        TR=args.TR, 
        trname=args.trname,
        n_features=args.n_features, 
        n_univariate=args.n_univariate, 
        n_accumulator=args.n_accumulator,
        n_decision=args.n_decision, 
        n_noise=args.n_noise, 
        drift_noise=args.drift_noise, 
        step_noise=args.step_noise,
        z_noise=args.z_noise,
        drift_noise_param=args.drift_noise_param,
        step_noise_param=args.step_noise_param, 
        z_noise_param=args.z_noise_param,
        noise_f=white, hrf_f=hrf_f, hrf_params=hrf_params, prng=prng)


