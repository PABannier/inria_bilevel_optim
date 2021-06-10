import numpy as np
import joblib

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


def solver(M, G, n_orient=1):
    alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max * 0.65, num=50)
    criterion = SUREForReweightedMultiTaskLasso(1, alphas, n_orient=n_orient)
    best_sure, _ = criterion.get_val(G, M)
    return best_sure


if __name__ == "__main__":
    sure_scores = []
    evoked_full = joblib.load("evokeds/sub-CC720670/evoked_fixed_full.pkl")

    fwd_fname = data_path / "meg" / f"{folder_name}_task-passive-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-passive-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-passive_cleaned-epo.fif"
    )

    signal = evoked_full.data

    for t in range(0, signal.shape[-1], 40):
        tmin = t / 10
        tmax = (t + 40) / 10
        print("Tmin:", tmin)
        print("Tmax:", tmax)

        evoked = evoked_full.crop(tmin=tmin, tmax=tmax)
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
        sure_scores.append(best_sure)

    print("Best SURE:", np.min(sure_scores))

    best_tmin = np.argmin(sure_scores) * 40
    best_tmax = (np.argmin(sure_scores) + 1) * 40
    print(f"Best window frame: ({best_tmin}, {best_tmax})")

    # Plotting optimal window frame
    fig = plt.figure()
    plt.plot(evoked_full.data)

    plt.axvline(
        best_tmin,
        linestyle="-",
        c="r",
        linewidth=1,
    )

    plt.axvline(
        best_tmax,
        linestyle="-",
        c="r",
        linewdith=1,
    )

    plt.show()
