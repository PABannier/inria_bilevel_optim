import numpy as np
from numpy.linalg import norm
import joblib

from pathlib import Path

import matplotlib.pyplot as plt
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.utils_datasets import compute_alpha_max

import mne

from mne.inverse_sparse.mxne_inverse import (
    _prepare_gain,
    is_fixed_orient,
    _reapply_source_weighting,
    _make_sparse_stc,
)
from numpy.core.fromnumeric import argmin

LOOSE = 0
STEP = 0.2

DATA_PATH = Path(
    "../../../../../rhochenb/Data/Cam-CAN/BIDS/derivatives/mne-study-template"
)

SUBJECT = "sub-CC110087"


def solver(M, G, n_orient=1):
    alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max * 0.65, num=15)
    criterion = SUREForReweightedMultiTaskLasso(1, alphas, n_orient=n_orient)
    best_sure, _ = criterion.get_val(G, M)
    return best_sure


if __name__ == "__main__":
    sure_scores = []
    evoked_full = joblib.load(f"evokeds/{SUBJECT}/evoked_fixed_full.pkl")

    data_path = DATA_PATH / SUBJECT
    folder_name = SUBJECT

    fwd_fname = data_path / "meg" / f"{folder_name}_task-passive-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-passive-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-passive_cleaned-epo.fif"
    )

    for t in np.arange(evoked_full.tmin, evoked_full.tmax, STEP):
        tmin = t
        tmax = tmin + STEP

        print("t_min:", tmin)
        print("t_max:", tmax)

        evoked = evoked_full.copy()
        evoked = evoked.crop(tmin=tmin, tmax=tmax)
        evoked = evoked.pick_types(eeg=False, meg=True)

        cleaned_epochs = mne.read_epochs(cleaned_epo_fname)
        noise_cov = mne.compute_covariance(cleaned_epochs, tmax=0, rank="info")

        forward = mne.read_forward_solution(fwd_fname)

        all_ch_names = evoked.ch_names

        (
            forward,
            gain,
            gain_info,
            whitener,
            source_weighting,
            _,
        ) = _prepare_gain(
            forward,
            evoked.info,
            noise_cov,
            pca=False,
            depth=0.8,
            loose=LOOSE,
            weights=None,
            weights_min=None,
            rank="info",
        )

        sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
        M = evoked.data[sel]

        M = np.dot(whitener, M)

        n_orient = 1 if is_fixed_orient(forward) else 3
        best_sure = solver(M, gain, n_orient)
        best_sure /= np.prod(M.shape)
        sure_scores.append(best_sure)

    print("Best SURE:", np.min(sure_scores))

    best_t_min = evoked_full.tmin + STEP * np.argmin(sure_scores)
    best_t_max = evoked_full.tmin + STEP * (np.argmin(sure_scores) + 1)

    print("Best t_min:", best_t_min)
    print("Best t max:", best_t_max)

    evoked_full_duration = evoked_full.tmax - evoked_full.tmin
    best_t_min_idx = (M.shape[1] * best_t_min) / evoked_full_duration
    best_t_max_idx = (M.shape[1] * best_t_max) / evoked_full_duration
    print(f"Best window frame: ({best_t_min_idx}, {best_t_max_idx})")

    # Plotting optimal window frame
    fig = plt.figure()
    plt.plot(evoked_full.data.mean(axis=-1), label="Average signal")

    plt.axvline(
        best_t_min_idx,
        linestyle="-",
        c="r",
        linewidth=1,
        label="Selected window frame",
    )

    plt.axvline(
        best_t_max_idx,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    plt.legend()
    plt.show()
