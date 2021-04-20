import os.path as op
import argparse
from tqdm import tqdm
import joblib

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

import mne
from mne.datasets import somato
from mne.viz import plot_sparse_source_estimates

from celer import MultiTaskLasso, MultiTaskLassoCV

from mtl.utils_datasets import compute_alpha_max
from mtl.sure import SURE
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.mtl import ReweightedMultiTaskLasso

parser = argparse.ArgumentParser()
parser.add_argument(
    "--estimator",
    help="choice of estimator to reconstruct the channels. available: "
    + "lasso-cv, lasso-sure, adaptive-cv, adaptive-sure",
)

args = parser.parse_args()
ESTIMATOR = args.estimator

mem = joblib.Memory(location=".")


@mem.cache
def load_data():
    data_path = somato.data_path()
    subject = "01"
    task = "somato"
    raw_fname = op.join(
        data_path,
        "sub-{}".format(subject),
        "meg",
        "sub-{}_task-{}_meg.fif".format(subject, task),
    )
    fwd_fname = op.join(
        data_path,
        "derivatives",
        "sub-{}".format(subject),
        "sub-{}_task-{}-fwd.fif".format(subject, task),
    )

    condition = "Unknown"

    # Read evoked
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.find_events(raw, stim_channel="STI 014")
    reject = dict(grad=4000e-13, eog=350e-6)
    picks = mne.pick_types(raw.info, meg=True, eog=True)

    event_id, tmin, tmax = 1, -1.0, 3.0
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        reject=reject,
        preload=True,
    )
    evoked = epochs.filter(1, None).average()
    evoked = evoked.pick_types(meg=True)
    evoked.crop(tmin=0.03, tmax=0.05)  # Choose a timeframe not too large

    # Compute noise covariance matrix
    cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, cov


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    from mne.inverse_sparse.mxne_inverse import (
        _prepare_gain,
        is_fixed_orient,
        _reapply_source_weighting,
        _make_sparse_stc,
        _make_dipoles_sparse,
    )
    from mne.inverse_sparse import make_stc_from_dipoles

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

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    # Whiten data
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

    # alphas = np.geomspace(alpha_max, alpha_max / 10, num=15)
    alphas = np.geomspace(alpha_max / 5, alpha_max / 100, num=30)
    n_folds = 5

    best_alpha_ = None

    if ESTIMATOR == "lasso-cv":
        # CV
        estimator = MultiTaskLassoCV(alphas=alphas, cv=n_folds)
        estimator.fit(G, M)
        best_alpha_ = estimator.alpha_

        # Refitting
        estimator = MultiTaskLasso(best_alpha_)
        estimator.fit(G, M)

        X = estimator.coef_.T

    elif ESTIMATOR == "lasso-sure":
        # SURE computation
        best_sure_ = np.inf

        for alpha in tqdm(alphas, total=len(alphas)):
            estimator = SURE(MultiTaskLasso, 1, random_state=0)
            sure_val_ = estimator.get_val(G, M, alpha)
            if sure_val_ < best_sure_:
                best_sure_ = sure_val_
                best_alpha_ = alpha

        print("best sure", best_sure_)
        # Refitting
        estimator = MultiTaskLasso(best_alpha_)
        estimator.fit(G, M)

        X = estimator.coef_.T

    elif ESTIMATOR == "adaptive-cv":
        # CV
        estimator = ReweightedMultiTaskLassoCV(alphas=alphas, n_folds=n_folds)
        estimator.fit(G, M)
        best_alpha_ = estimator.best_alpha_

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha_)
        estimator.fit(G, M)

        X = estimator.coef_

    elif ESTIMATOR == "adaptive-sure":
        # SURE computation
        best_sure_ = np.inf

        for alpha in tqdm(alphas, total=len(alphas)):
            estimator = SURE(ReweightedMultiTaskLasso, 1, random_state=0)
            sure_val_ = estimator.get_val(G, M, alpha)
            if sure_val_ < best_sure_:
                best_sure_ = sure_val_
                best_alpha_ = alpha

        print("best sure", best_sure_)
        print("best alpha", best_alpha_)

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha_)
        estimator.fit(G, M)

        X = estimator.coef_

    else:
        raise ValueError(
            "Invalid estimator. Please choose between "
            + "lasso-cv, lasso-sure, adaptive-cv or adaptive-sure"
        )

    active_set = norm(X, axis=1) != 0
    return X[active_set, :], active_set


if __name__ == "__main__":
    loose, depth = 0, 0.9  # Fixed orientation
    evoked, forward, noise_cov = load_data()

    stc = apply_solver(solver, evoked, forward, noise_cov, loose, depth)

    plot_sparse_source_estimates(
        forward["src"],
        stc,
        bgcolor=(1, 1, 1),
        opacity=0.1,
        fig_name="Source activations",
    )
