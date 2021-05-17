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


def load_data(folder_name, data_path):
    data_path = Path(data_path)

    subject_dir = data_path / "subjects"

    fwd_fname = data_path / "meg" / f"{folder_name}_task-task-fwd.fif"
    ave_fname = data_path / "meg" / f"{folder_name}_task-task-ave.fif"
    cleaned_epo_fname = (
        data_path / "meg" / f"{folder_name}_task-task_cleaned-epo.fif"
    )

    # Building noise covariance
    cleaned_epochs = mne.read_epochs(cleaned_epo_fname)
    noise_cov = mne.compute_covariance(cleaned_epochs, tmax=0)

    evoked = mne.read_evokeds(ave_fname, condition=None, baseline=(None, 0))[
        0
    ]  # Use evoked.plot()
    evoked.crop(tmin=0.05, tmax=0.1)
    evoked = evoked.pick_types(eeg=False, meg=True)

    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov, subject_dir


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
    alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max / 10, num=15)

    start = time.time()

    criterion = SUREForReweightedMultiTaskLasso(1, alphas, n_orient=n_orient)
    best_sure, best_alpha = criterion.get_val(G, M)

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

    evoked, forward, noise_cov, subject_dir = load_data(folder_name, data_path)

    stc, residual = apply_solver(
        solver, evoked, forward, noise_cov, loose, depth
    )

    print("=" * 20)
    print("Explained variance:", norm(residual.data) / norm(evoked.data))
    print("=" * 20)

    return stc, residual, evoked, noise_cov, subject_dir


###################################
######### GENERATE REPORT #########
###################################


def add_foci_to_brain_surface(brain, stc):
    fig, ax = plt.subplots(figsize=(10, 4))

    for i_hemi, hemi in enumerate(["lh", "rh"]):
        surface_coords = brain.geo[hemi].coords
        hemi_data = stc.lh_data if hemi == "lh" else stc.rh_data
        for k in range(len(stc.vertices[i_hemi])):
            activation_idx = stc.vertices[i_hemi][k]
            foci_coords = surface_coords[activation_idx]

            (line,) = ax.plot(stc.times, 1e9 * hemi_data[k])
            brain.add_foci(foci_coords, hemi=hemi, color=line.get_color())

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (nAm)")

    return fig


def generate_report(
    patient_id, out_path, stc, evoked, residual, noise_cov, subject_dir
):
    title = patient_id
    report = mne.report.Report(title=title)

    views = ["lat", "med"]

    kwargs = dict(
        views=views,
        hemi="split",
        subjects_dir=subject_dir,
        initial_time=0.0,
        clim="auto",
        colorbar=False,
        show_traces=False,
        time_viewer=False,
        cortex="low_contrast",
    )

    brain = stc.plot(**kwargs)
    fig_traces = add_foci_to_brain_surface(brain, stc)

    # brain.toggle_interface(False)
    fig = brain.screenshot(time_viewer=True)
    brain.close()
    exp_var = norm(residual.data) / norm(evoked.data)

    evoked_fig = evoked.plot(ylim=dict(mag=[-250, 250], grad=[-100, 100]))
    residual_fig = residual.plot(ylim=dict(mag=[-250, 250], grad=[-100, 100]))

    evoked_fig_white = evoked.plot_white(noise_cov=noise_cov)
    residual_fig_white = residual.plot_white(noise_cov=noise_cov)

    report.add_figs_to_section(evoked_fig, "Evoked", section="Sensor")
    report.add_figs_to_section(residual_fig, "Residual", section="Sensor")

    report.add_figs_to_section(
        evoked_fig_white, "Evoked - White noise", section="Sensor"
    )
    report.add_figs_to_section(
        residual_fig_white, "Residual - White noise", section="Sensor"
    )

    report.add_figs_to_section(
        fig,
        f"Source estimate, explained variance: {exp_var:.2f}",
        section="Source",
    )
    report.add_figs_to_section(
        fig_traces, "Source Time Courses", section="Source"
    )

    report.add_figs_to_section(sure_path_fig, "SURE Path", section="Source")

    report.save(out_path)
