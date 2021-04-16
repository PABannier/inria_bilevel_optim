import joblib
from tqdm import tqdm
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from numpy.linalg import norm

import mne
from mne.datasets import sample

from celer import MultiTaskLassoCV, MultiTaskLasso

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.utils_datasets import compute_alpha_max, plot_sure_mse_path
from mtl.sure import SURE


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
    subjects_dir = data_path + "/subjects"

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # Handling average file
    evoked = mne.read_evokeds(
        ave_fname, condition="Left Auditory", baseline=(None, 0)
    )
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked = evoked.pick_types(eeg=False, meg=True)
    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    from mne.inverse_sparse.mxne_inverse import (
        _prepare_gain,
        is_fixed_orient,
        _reapply_source_weighting,
        _make_sparse_stc,
    )

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward,
        evoked.info,
        noise_cov,
        pca=False,
        depth=depth,
        loose=loose,
        weights=None,
        weights_min=None,
        rank=None,
    )

    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    M = np.dot(whitener, M)

    n_orient = 1 if is_fixed_orient(forward) else 3
    X, active_set = solver(M, gain, n_orient)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(
        X,
        active_set,
        forward,
        tmin=evoked.times[0],
        tstep=1.0 / evoked.info["sfreq"],
    )

    return stc


def solver(M, G, n_orient=1):
    alpha_max = compute_alpha_max(G, M)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max / 10, num=50)
    n_folds = 5

    best_alpha_ = None

    start = time.time()

    estimator = ReweightedMultiTaskLassoCV(
        alphas=alphas, n_folds=n_folds, warm_start=True
    )
    estimator.fit(G, M)
    best_alpha_ = estimator.best_alpha_

    # Refitting
    estimator = ReweightedMultiTaskLasso(best_alpha_, warm_start=True)
    estimator.fit(G, M)

    print("Time", time.time() - start)

    X = estimator.coef_

    active_set = norm(X, axis=1) != 0
    return X[active_set, :], active_set


if __name__ == "__main__":
    loose, depth = 0, 0.9  # Fixed orientation
    evoked, forward, noise_cov = load_data()

    start = time.time()
    stc = apply_solver(solver, evoked, forward, noise_cov, loose, depth)
