import pytest
import numpy as np
from fmrilearnexp.common import extract_trial_features
from fmrilearnexp.common import load_dimreduce_data_fromcl
from fmrilearn.preprocess.labels import create_y


def test_load_dimreduce_data_fromcl():
    csvs = [
            "/data/data2/meta_accumulate//fh/mvpa/fh_pca_space_merge_Insula_rt_fast.csv",
            "/data/data2/meta_accumulate//fh/mvpa/fh_pca_space_merge_Insula_rt_slow.csv"
            ]

    # load_dimreduce_data_fromcl(csvs, feature_cols, label_col, cv_col, trial_tr_col)
    feature_col = range(0,3)
    label_col = 5
    cv_col = 6
    trial_tr_col = 7
    Xs, ys, indices, cvcodes = load_dimreduce_data_fromcl(
            csvs, feature_col, label_col, cv_col, trial_tr_col
            )
    assert len(Xs) == 2, "Xs wrong len"
    assert len(ys) == 2, "ys wrong len"
    assert len(indices) == 2, "indices wrong len"
    assert len(cvcodes) == 2, "cvcodes wrong len"
    assert Xs[0].shape == (210, 3), "X0 wrong shape"
    assert Xs[1].shape == (210, 3), "X1 wrong shape"
    assert Xs[0].shape[0] == ys[0].shape[0], "X0 y0 mistmatch"
    assert Xs[1].shape[0] == ys[1].shape[0], "X0 y0 mistmatch"
    assert ys[0].shape == indices[0].shape, "y0 i0 mistmatch"
    assert ys[1].shape == indices[1].shape, "y1 i1 mistmatch"
    assert ys[0].shape == cvcodes[0].shape, "y0 cv0 mistmatch"
    assert ys[1].shape == cvcodes[1].shape, "y1 cv1 mistmatch"

    print("X0, the first few rows\n{0}".format(Xs[0][:5,:]))
    print("y0, the first few elements {0}".format(ys[0][:5]))
    print("index0, the first few elements {0}".format(indices[0][:5]))
    print("cv0, the first few elements {0}".format(cvcodes[0][:5]))
    
    
def test_extract_trial_features_simulated():    
    prng = np.random.RandomState(43)
    
    # Create data
    iti = 2
    updown = range(1,4) + range(-2,0)
    l = len(updown)
    accum2 = np.asarray([0]*iti + updown + [0]*iti + updown)
    nullfea = prng.rand(accum2.shape[0])
    X = np.vstack([accum2, nullfea]).T
    
    y = np.asarray([0]*iti + [1]*l + [0]*iti + [2]*l)
    trials = np.asarray([0]*iti + [1]*l + [0]*iti + [2]*l)
    assert y.shape == trials.shape, "y and trials do not match"
    assert X.shape[0] == y.shape[0], "y and X do not match" 

    print("\n*** Test 1: Full window ***")
    window=range(l-1)
    Xfea, indexfea, otherfea = extract_trial_features(
            X, trials, window, [trials], 0
            )
    yea = otherfea[0]

    assert Xfea.shape[0] == 2, "Xfea nrow is wrong"
    assert Xfea.shape[1] == 2 * 5, "Xfea wrong feature number"
    assert len(yea) == 2, "yfea is off"
    assert len(indexfea) == 2, "trialfea is off"

    # Xmax, Xmin, Xdiff, Xmean, Xslope
    print("Xmax\n{0}".format(Xfea[:,0:2]))
    print("Xmin\n{0}".format(Xfea[:,2:4]))
    print("Xdiff\n{0}".format(Xfea[:,4:6]))
    print("Xmean\n{0}".format(Xfea[:,6:8]))
    print("Xslope\n{0}".format(Xfea[:,8:10]))


def test_extract_trial_features_real():
    csvs = [
            "./data/fh_pca_space_merge_Insula_rt_fast.csv",
            "./data/fh_pca_space_merge_Insula_rt_slow.csv"
            ]

    # load_dimreduce_data_fromcl(csvs, feature_cols, label_col, cv_col, trial_tr_col)
    feature_col = range(0,3)
    label_col = 5
    cv_col = 6
    trial_tr_col = 7
    Xs, ys, indices, cvcodes = load_dimreduce_data_fromcl(
            csvs, feature_col, label_col, cv_col, trial_tr_col
            )
    
    X = np.concatenate(Xs)
    y = create_y(np.concatenate(ys)) 
    trials = np.concatenate(indices)
    windowstr = '0:10'
    window =  range(*[int(i) for i in windowstr.split(':')])

    Xfea, indexfea, otherfea = extract_trial_features(
            X, trials, window, [trials], None
            )
    yea = otherfea[0]
    
    # Xmax, Xmin, Xdiff, Xmean, Xslope
    print("Xmax\n{0}".format(Xfea[:,0:3]))
    print("Xmin\n{0}".format(Xfea[:,3:6]))
    print("Xdiff\n{0}".format(Xfea[:,6:9]))
    print("Xmean\n{0}".format(Xfea[:,9:12]))
    print("Xslope\n{0}".format(Xfea[:,12:15]))
    

