import joblib
import os
import glob

import numpy as np
from numpy.linalg import norm

import pandas as pd

import matplotlib.pyplot as plt

import mne
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.mxne_inverse import _compute_residual

WORKING_EXAMPLES = [
    "CC720670",
    "CC320461",
    "CC420004",
    "CC520200",
    # "CC720188",
    "CC110033",
]


PATIENT = "CC320687"
ORIENTATION = "fixed"


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


def generate_brain_plot(stc_fixed, stc_free):
    views = ["lat", "med"]

    kwargs = dict(
        views=views,
        hemi="split",
        subjects_dir="subjects_dir",
        initial_time=0.0,
        clim="auto",
        colorbar=False,
        show_traces=False,
        time_viewer=False,
        cortex="low_contrast",
    )

    brain_fixed = stc_fixed.plot(**kwargs)
    fig_traces_fixed = add_foci_to_brain_surface(brain_fixed, stc_fixed)
    fig_fixed = brain_fixed.screenshot(time_viewer=True)
    brain_fixed.close()

    brain_free = stc_free.plot(**kwargs)
    fig_traces_free = add_foci_to_brain_surface(brain_free, stc_free)
    fig_free = brain_free.screenshot(time_viewer=True)
    brain_free.close()

    return fig_fixed, fig_free, fig_traces_fixed, fig_traces_free


def generate_residual_plots(evoked, evoked_full, residual, noise_cov):
    evoked_fig = evoked.plot(
        ylim=dict(mag=[-250, 250], grad=[-100, 100]), spatial_colors=True
    )
    residual_fig = residual.plot(
        ylim=dict(mag=[-250, 250], grad=[-100, 100]), spatial_colors=True
    )

    topo_fig = evoked.plot_topomap(ch_type="grad")

    evoked_full_fig = evoked_full.plot(spatial_colors=True)
    ax_list = evoked_full_fig.axes
    for i in range(2):
        ax_list[i].axvline(0.08, linestyle="--", linewidth=2, c="r")
        ax_list[i].axvline(
            0.15,
            linestyle="--",
            linewidth=2,
            c="r",
            label="fitted window frame",
        )

        ax_list[i].legend()

    evoked_fig_white = evoked.plot_white(noise_cov=noise_cov)
    residual_fig_white = residual.plot_white(noise_cov=noise_cov)

    evoked_full_fig_white = evoked_full.plot_white(noise_cov=noise_cov)

    return (
        evoked_fig,
        topo_fig,
        evoked_full_fig,
        residual_fig,
        evoked_fig_white,
        residual_fig_white,
        evoked_full_fig_white,
    )


def plot_sure_path(alphas, sure_path):
    # Saving SURE path for better visualization in reports
    fig = plt.figure()

    plt.semilogx(
        alphas / np.max(alphas),
        sure_path,
        label="Path",
        marker="x",
        markeredgecolor="red",
    )
    plt.title("Sure path", fontweight="bold", fontsize=16)

    plt.axvline(
        alphas[np.argmin(sure_path)] / np.max(alphas),
        linestyle="--",
        linewidth=2,
        label="best $\lambda$ - SURE",
        c="r",
    )

    plt.legend()

    plt.xlabel("$\lambda / \lambda_{max}$")
    plt.ylabel("SURE")

    return fig


def generate_report(figs):
    report = mne.report.Report(title="CamCan - SURE Reweighted MTL")

    for k, v in figs.items():
        (
            fig_fixed,
            fig_free,
            fig_traces_fixed,
            fig_traces_free,
            fig_topomap,
            fig_evoked_fixed,
            fig_evoked_free,
            fig_evoked_fixed_full,
            fig_evoked_free_full,
            fig_residual_fixed,
            fig_residual_free,
            fig_evoked_white_fixed,
            fig_evoked_white_free,
            fig_evoked_full_white,
            fig_residual_white_fixed,
            fig_residual_white_free,
            fig_sure_path_fixed,
            fig_sure_path_free,
            age,
        ) = v

        k = k + f" ({str(age)})" if age else k

        report.add_figs_to_section(fig_fixed, k + " - Fixed", section=k)
        report.add_figs_to_section(
            fig_traces_fixed, k + " - Fixed - Activation", section=k
        )
        report.add_figs_to_section(fig_free, k + " - Free", section=k)
        report.add_figs_to_section(
            fig_traces_free, k + " - Free - Activation", section=k
        )

        report.add_figs_to_section(
            fig_evoked_fixed_full, k + " - Evoked (full)", section=k
        )
        report.add_figs_to_section(fig_evoked_free, k + " - Evoked", section=k)

        report.add_figs_to_section(fig_topomap, k + " - Topomap", section=k)

        report.add_figs_to_section(
            fig_residual_fixed, k + " - Residual - Fixed", section=k
        )
        report.add_figs_to_section(
            fig_residual_free, k + " - Residual - Free", section=k
        )

        report.add_figs_to_section(
            fig_evoked_white_fixed,
            k + " - Evoked - White noise",
            section=k,
        )

        report.add_figs_to_section(
            fig_evoked_full_white,
            k + " - Evoked (full) - White noise",
            section=k,
        )

        report.add_figs_to_section(
            fig_residual_white_fixed,
            k + " - Residual - White noise - Fixed",
            section=k,
        )
        report.add_figs_to_section(
            fig_residual_white_free,
            k + " - Residual - White noise - Free",
            section=k,
        )

        report.add_figs_to_section(
            fig_sure_path_fixed, k + " - SURE Path - Fixed", section=k
        )
        report.add_figs_to_section(
            fig_sure_path_free, k + " - SURE Path - Free", section=k
        )

    report.save(f"reports/report_youngest_3.html", overwrite=True)


if __name__ == "__main__":
    figs = {}

    # paths = glob.glob("stcs/*")
    # ids = [x.split("/")[-1][4:] for x in paths]

    participant_info_df = pd.read_csv("participants.tsv", sep="\t")
    participant_info_df = participant_info_df[participant_info_df["age"] < 30]

    participant_list = list(participant_info_df["participant_id"])
    participant_list = [x[4:] for x in participant_list][:6]

    for subject_id in participant_list:
        print(subject_id)

        # Load evoked, noise_cov, residual, sure_path
        evoked_free_full = joblib.load(
            f"evokeds/sub-{subject_id}/evoked_free_full.pkl"
        )
        evoked_fixed_full = joblib.load(
            f"evokeds/sub-{subject_id}/evoked_fixed_full.pkl"
        )

        evoked_free = joblib.load(f"evokeds/sub-{subject_id}/evoked_free.pkl")
        evoked_fixed = joblib.load(
            f"evokeds/sub-{subject_id}/evoked_fixed.pkl"
        )

        noise_cov_free = joblib.load(
            f"noise_covs/sub-{subject_id}/noise_cov_free.pkl"
        )
        noise_cov_fixed = joblib.load(
            f"noise_covs/sub-{subject_id}/noise_cov_fixed.pkl"
        )

        residual_free = joblib.load(
            f"residuals/sub-{subject_id}/residual_free.pkl"
        )
        residual_fixed = joblib.load(
            f"residuals/sub-{subject_id}/residual_fixed.pkl"
        )

        alphas_free = joblib.load(
            f"sure_paths/sub-{subject_id}/alphas_free.pkl"
        )
        alphas_fixed = joblib.load(
            f"sure_paths/sub-{subject_id}/alphas_fixed.pkl"
        )

        sure_path_free = joblib.load(
            f"sure_paths/sub-{subject_id}/sure_path_free.pkl"
        )
        sure_path_fixed = joblib.load(
            f"sure_paths/sub-{subject_id}/sure_path_fixed.pkl"
        )

        stc_free = joblib.load(f"stcs/sub-{subject_id}/free.pkl")
        stc_fixed = joblib.load(f"stcs/sub-{subject_id}/fixed.pkl")

        subjects_dir = "subjects_dir"

        try:
            (
                fig_fixed,
                fig_free,
                fig_traces_fixed,
                fig_traces_free,
            ) = generate_brain_plot(stc_fixed, stc_free)
        except IndexError:
            print(f"No sources found for {subject_id}")
            continue

        (
            fig_evoked_fixed,
            fig_topomap,
            fig_evoked_fixed_full,
            fig_residual_fixed,
            fig_evoked_white_fixed,
            fig_residual_white_fixed,
            fig_evoked_full_white,
        ) = generate_residual_plots(
            evoked_fixed,
            evoked_fixed_full,
            residual_fixed,
            noise_cov_fixed,
        )

        (
            fig_evoked_free,
            _,
            fig_evoked_free_full,
            fig_residual_free,
            fig_evoked_white_free,
            fig_residual_white_free,
            _,
        ) = generate_residual_plots(
            evoked_free, evoked_free_full, residual_free, noise_cov_free
        )

        fig_sure_path_fixed = plot_sure_path(alphas_fixed, sure_path_fixed)
        fig_sure_path_free = plot_sure_path(alphas_free, sure_path_free)

        age = participant_info_df[
            participant_info_df["participant_id"] == f"sub-{subject_id}"
        ]["age"]

        figs[subject_id] = (
            fig_fixed,
            fig_free,
            fig_traces_fixed,
            fig_traces_free,
            fig_topomap,
            fig_evoked_fixed,
            fig_evoked_free,
            fig_evoked_fixed_full,
            fig_evoked_free_full,
            fig_residual_fixed,
            fig_residual_free,
            fig_evoked_white_fixed,
            fig_evoked_white_free,
            fig_evoked_full_white,
            fig_residual_white_fixed,
            fig_residual_white_free,
            fig_sure_path_fixed,
            fig_sure_path_free,
            int(age),
        )

    generate_report(figs)
