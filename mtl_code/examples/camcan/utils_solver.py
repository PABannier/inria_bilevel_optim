import os
from pathlib import Path
import joblib
import time
from tqdm import tqdm
import os.path as op

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

import mne
from mne.inverse_sparse.mxne_inverse import _compute_residual

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.utils_datasets import compute_alpha_max


def load_data(folder_name, data_path, orient):
    data_path = Path(data_path)

    subject_dir = data_path / "subjects"

    fwd_fname = data_path / "meg" / f"{folder_name}_task-passive-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-passive-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-passive_cleaned-epo.fif"
    )

    # Building noise covariance
    cleaned_epochs = mne.read_epochs(cleaned_epo_fname)
    noise_cov = mne.compute_covariance(cleaned_epochs, tmax=0, rank="info")

    evokeds = mne.read_evokeds(ave_fname, condition=None, baseline=(None, 0))
    evoked = evokeds[-2]

    if not os.path.exists(f"evokeds/{folder_name}"):
        os.mkdir(f"evokeds/{folder_name}")
    joblib.dump(evoked, f"evokeds/{folder_name}/evoked_{orient}_full.pkl")

    evoked.crop(tmin=0.08, tmax=0.15)  # 0.08 - 0.15
    evoked = evoked.pick_types(eeg=False, meg=True)

    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov, subject_dir


def apply_solver(
    solver, evoked, forward, noise_cov, folder_name, loose=0.2, depth=0.8
):
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
        rank="info",
    )

    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    M = np.dot(whitener, M)

    orient = "fixed" if loose == 0 else "free"

    # Save evoked_full_whitened
    if not os.path.exists(f"evokeds/{folder_name}"):
        os.mkdir(f"evokeds/{folder_name}")
    joblib.dump(M, f"evokeds/{folder_name}/evoked_{orient}_full_whitened.pkl")

    n_orient = 1 if is_fixed_orient(forward) else 3
    print("=" * 20)
    print("Number of orientations:", n_orient)
    print("=" * 20)
    X, active_set = solver(M, gain, folder_name, n_orient)
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


def solver(M, G, folder_name, n_orient=1):
    alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max * 0.65, num=50)

    start = time.time()

    criterion = SUREForReweightedMultiTaskLasso(1, alphas, n_orient=n_orient)
    best_sure, best_alpha = criterion.get_val(G, M)

    orient = "fixed" if n_orient == 1 else "free"

    if not os.path.exists(f"sure_paths/{folder_name}"):
        os.mkdir(f"sure_paths/{folder_name}")

    joblib.dump(alphas, f"sure_paths/{folder_name}/alphas_{orient}.pkl")
    joblib.dump(
        criterion.sure_path_,
        f"sure_paths/{folder_name}/sure_path_{orient}.pkl",
    )

    print("Duration:", time.time() - start)

    print("Best SURE:", best_sure)
    print("Best alpha:", best_alpha)

    # Refitting
    estimator = ReweightedMultiTaskLasso(best_alpha, n_orient=n_orient)
    estimator.fit(G, M)

    X = estimator.coef_
    active_set = norm(X, axis=1) != 0

    return X[active_set, :], active_set


def solve_inverse_problem(folder_name, data_path, loose, depth=0.9):
    """Solves an inverse problem using SURE Reweighted MTL

    Parameters
    ----------
    folder_name: str
        The unique ID of the patient folder.

    data_path: str
        The data location.

    loose: float
        0 for fixed orientation. 0.9 for free orientation.

    depth: float, default=0.9
        The exponent that raises the norm used to normalize sources.
    """

    orient = "fixed" if loose == 0 else "free"

    evoked, forward, noise_cov, subject_dir = load_data(
        folder_name, data_path, orient
    )

    stc, residual = apply_solver(
        solver, evoked, forward, noise_cov, folder_name, loose, depth
    )

    if not os.path.exists(f"evokeds/{folder_name}"):
        os.mkdir(f"evokeds/{folder_name}")
    joblib.dump(evoked, f"evokeds/{folder_name}/evoked_{orient}.pkl")

    if not os.path.exists(f"noise_covs/{folder_name}"):
        os.mkdir(f"noise_covs/{folder_name}")
    joblib.dump(noise_cov, f"noise_covs/{folder_name}/noise_cov_{orient}.pkl")

    if not os.path.exists(f"residuals/{folder_name}"):
        os.mkdir(f"residuals/{folder_name}")
    joblib.dump(residual, f"residuals/{folder_name}/residual_{orient}.pkl")

    print("=" * 20)
    print("Explained variance:", norm(residual.data) / norm(evoked.data))
    print("=" * 20)

    return stc, residual, evoked, noise_cov, subject_dir, forward
