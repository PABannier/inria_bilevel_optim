import argparse
import joblib
import time
from pathlib import Path
from tqdm import tqdm
import os.path as op

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from joblib import Memory

import mne
from mne.datasets import sample, somato
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.mxne_inverse import _compute_residual

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure import SURE
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.utils_datasets import compute_alpha_max

parser = argparse.ArgumentParser()
parser.add_argument("--condition", help="condition")
parser.add_argument("--dataset", help="dataset")

args = parser.parse_args()

ALPHA_MAX, ALPHA_MIN = 0, 0

if args.dataset is None:
    raise ValueError(
        "Please specify a dataset by using --dataset argument. "
        + "Available dataset: sample, somato."
    )
elif args.dataset == "sample":
    if args.condition is None:
        raise ValueError(
            "Please specify a regularizing constant by using --condition argument. "
            + "Available condition: Left Auditory, Right Auditory, Left visual, Right visual."
        )


# mem = Memory(".")


def load_data():

    if args.dataset == "sample":
        data_path = sample.data_path()
        fwd_fname = (
            data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
        )
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
    else:
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

        subjects_dir = op.join(
            data_path, "derivatives", "freesurfer", "subjects"
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
        # max_t = evoked.get_peak()[1]

        # Handling forward solution
        forward = mne.read_forward_solution(fwd_fname)
        noise_cov = mne.compute_covariance(epochs, rank="info", tmax=0.0)

    return (
        evoked,
        forward,
        noise_cov,
        subjects_dir,
    )


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    """Call a custom solver on evoked data.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.

    Parameters
    ----------
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
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

    residual = _compute_residual(forward, evoked, X, active_set, gain_info)

    return stc, residual


def solver(M, G, n_orient=1):
    """Run L2 penalized regression and keep 10 strongest locations.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    alpha_max = compute_alpha_max(G, M, n_orient=n_orient)
    print("Alpha max:", alpha_max)

    alphas = np.geomspace(alpha_max, alpha_max / 10, num=15)

    global ALPHA_MAX
    global ALPHA_MIN

    ALPHA_MAX = alpha_max
    ALPHA_MIN = alpha_max / 10

    start = time.time()

    criterion = SUREForReweightedMultiTaskLasso(
        1, alphas, n_orient=n_orient, random_state=1
    )
    best_sure, best_alpha = criterion.get_val(G, M)

    # Saving SURE path for better visualization in reports
    if args.dataset == "sample":
        file_name = args.condition.lower().replace(" ", "_")
        out_path = f"data/sure_path_{file_name}.pkl"
    elif args.dataset == "somato":
        out_path = f"data/sure_path_somato.pkl"

    joblib.dump(criterion.sure_path_, out_path)
    print(criterion.sure_path_)

    print("Duration:", time.time() - start)

    print("Best SURE:", best_sure)
    print("Best alpha:", best_alpha)

    # Refitting
    estimator = ReweightedMultiTaskLasso(best_alpha, n_orient=n_orient)
    estimator.fit(G, M)

    X = estimator.coef_
    active_set = norm(X, axis=1) != 0

    return X[active_set, :], active_set


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


def plot_sure_path():
    # Saving SURE path for better visualization in reports
    if args.dataset == "sample":
        file_name = args.condition.lower().replace(" ", "_")
        out_path = f"data/sure_path_{file_name}.pkl"
    elif args.dataset == "somato":
        out_path = f"data/sure_path_somato.pkl"

    sure_path = joblib.load(out_path)

    fig = plt.figure()

    alphas = np.geomspace(ALPHA_MAX, ALPHA_MIN, num=15)
    plt.semilogx(alphas / np.max(alphas), sure_path, label="Path")
    plt.title("Sure path", fontweight="bold", fontsize=16)

    plt.axvline(
        alphas[np.argmin(sure_path)] / np.max(alphas),
        linestyle="--",
        linewidth=2,
        label="best $\lambda$ - SURE",
    )

    plt.legend()

    plt.xlabel("$\lambda / \lambda_{max}$")
    plt.ylabel("SURE")

    return fig


def generate_report(
    evoked_fig,
    evoked_fig_white,
    residual_fig,
    residual_fig_white,
    stc,
    sure_path_fig,
):
    title = args.condition if args.dataset == "sample" else "somato"
    report = mne.report.Report(title=title)

    views = ["lat", "med"]

    kwargs = dict(
        views=views,
        hemi="split",
        subjects_dir=subjects_dir,
        # initial_time=0.1,
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

    filename = (
        args.condition.lower().replace(" ", "_")
        if args.dataset == "sample"
        else "somato"
    )

    report.save(f"reports/report_{filename}.html", overwrite=True)


if __name__ == "__main__":
    loose, depth = 0.9, 0.9  # Free orientation
    (
        evoked,
        forward,
        noise_cov,
        subjects_dir,
    ) = load_data()

    if args.dataset == "sample":
        file_name = args.condition.lower().replace(" ", "_")
        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True, parents=True)
        stc_filepath = data_folder / f"stc_{file_name}.pkl"
        residual_filepath = data_folder / f"residual_{file_name}.pkl"
    else:
        folder = "somato"
        data_folder = Path("data")
        stc_filepath = data_folder / "stc_somato.pkl"
        residual_filepath = data_folder / "residual_somato.pkl"

    # if op.exists(stc_filepath) and op.exists(residual_filepath):
    #     stc = joblib.load(stc_filepath)
    #     residual = joblib.load(residual_filepath)
    # else:
    stc, residual = apply_solver(
        solver, evoked, forward, noise_cov, loose, depth
    )

    joblib.dump(stc, stc_filepath)
    joblib.dump(residual, residual_filepath)

    if args.condition == "Left visual":
        evoked_fig = evoked.plot(ylim=dict(mag=[-250, 250], grad=[-100, 100]))
        residual_fig = residual.plot(
            ylim=dict(mag=[-250, 250], grad=[-100, 100])
        )
    else:
        evoked_fig = evoked.plot(ylim=dict(mag=[-500, 500], grad=[-200, 200]))
        residual_fig = residual.plot(
            ylim=dict(mag=[-500, 500], grad=[-200, 200])
        )

    evoked_fig_white = evoked.plot_white(noise_cov=noise_cov)
    residual_fig_white = residual.plot_white(noise_cov=noise_cov)

    sure_path_fig = plot_sure_path()

    generate_report(
        evoked_fig,
        evoked_fig_white,
        residual_fig,
        residual_fig_white,
        stc,
        sure_path_fig,
    )
