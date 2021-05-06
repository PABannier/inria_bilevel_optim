import argparse
import joblib
from tqdm import tqdm
import os.path as op

import numpy as np
from numpy.linalg import norm

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.mxne_inverse import _compute_residual

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    help="regularizing hyperparameter",
)
parser.add_argument("--condition", help="condition")

args = parser.parse_args()

if args.condition is None:
    raise ValueError(
        "Please specify a regularizing constant by using --condition argument. "
        + "Available condition: Left Auditory, Right Auditory, Left visual, Right visual."
    )


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
    trans_fname = op.join(
        data_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    subjects_dir = data_path + "/subjects"

    bem_fname = op.join(
        subjects_dir, "sample", "bem", "sample-5120-bem-sol.fif"
    )

    condition = args.condition

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # Handling average file
    evoked = mne.read_evokeds(
        ave_fname, condition=condition, baseline=(None, 0)
    )
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked = evoked.pick_types(eeg=False, meg=True)
    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return (
        evoked,
        forward,
        noise_cov,
        cov_fname,
        bem_fname,
        trans_fname,
        subjects_dir,
    )


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    from mne.inverse_sparse.mxne_inverse import (
        _prepare_gain,
        is_fixed_orient,
        _reapply_source_weighting,
        _make_sparse_stc,
    )

    all_ch_names = evoked.ch_names

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
    print("=" * 20)
    print("Number of orientations:", n_orient)
    print("=" * 20)
    X, active_set = solver(M, gain, n_orient)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(
        X,
        active_set,
        forward,
        tmin=evoked.times[0],
        tstep=1.0 / evoked.info["sfreq"],
    )

    residual = _compute_residual(forward, evoked, X, active_set, gain_info)

    return stc, residual


def solver(M, G, n_orient=1):
    if args.alpha:
        estimator = ReweightedMultiTaskLasso(
            float(args.alpha), n_orient=n_orient
        )
        estimator.fit(G, M)
    else:
        alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
        print("Alpha max:", alpha_max)

        alphas = np.geomspace(alpha_max, alpha_max / 10, num=15)

        import time

        start = time.time()

        criterion = SUREForReweightedMultiTaskLasso(
            1, alphas, n_orient=n_orient
        )
        best_sure, best_alpha = criterion.get_val(G, M)

        print("Duration:", time.time() - start)

        print("Best SURE:", best_sure)
        print("Best alpha:", best_alpha)

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha, n_orient)
        estimator.fit(G, M)

    X = estimator.coef_
    active_set = norm(X, axis=1) != 0

    return X[active_set, :], active_set


if __name__ == "__main__":
    loose, depth = 0.9, 0.9
    (
        evoked,
        forward,
        noise_cov,
        cov_fname,
        bem_fname,
        trans_fname,
        subjects_dir,
    ) = load_data()

    stc, residual = apply_solver(
        solver, evoked, forward, noise_cov, loose, depth
    )

    print("=" * 20)
    print("Explained variance:", norm(residual.data) / norm(evoked.data))
    print("=" * 20)

    plot_sparse_source_estimates(
        forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1
    )
