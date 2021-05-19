import joblib
import os
import glob

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

import mne
from mne.viz import plot_sparse_source_estimates

WORKING_EXAMPLES = [
    "CC720670",
    "CC320461",
    "CC420004",
    "CC520200",
    "CC720188",
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


def generate_report(figs):
    report = mne.report.Report(title="CamCan - SURE Reweighted MTL")

    for k, v in figs.items():
        fig_fixed, fig_free, fig_traces_fixed, fig_traces_free = v
        report.add_figs_to_section(fig_fixed, k + " - Fixed", section=k)
        report.add_figs_to_section(
            fig_traces_fixed, k + " - Fixed - Activation", section=k
        )
        report.add_figs_to_section(fig_free, k + " - Free", section=k)
        report.add_figs_to_section(
            fig_traces_free, k + " - Free - Activation", section=k
        )

    report.save(f"reports/report_free_fixed.html", overwrite=True)


if __name__ == "__main__":
    figs = {}

    paths = glob.glob("stcs/*")
    ids = [x.split("/")[-1][4:] for x in paths]

    for subject_id in WORKING_EXAMPLES:
        print(subject_id)
        path_free = f"stcs/sub-{subject_id}/free.pkl"
        path_fixed = f"stcs/sub-{subject_id}/fixed.pkl"

        if os.path.exists(path_free) and os.path.exists(path_fixed):
            stc_free = joblib.load(path_free)
            stc_fixed = joblib.load(path_fixed)
        else:
            continue

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

        figs[subject_id] = (
            fig_fixed,
            fig_free,
            fig_traces_fixed,
            fig_traces_free,
        )

    generate_report(figs)

    # # Loading data
    # in_file = f"stcs/sub-{PATIENT}/{ORIENTATION}.pkl"
    # stc = joblib.load(in_file)

    # subjects_dir = "subjects_dir"

    # # Brain morphing
    # morph = mne.compute_source_morph(
    #     stc,
    #     subject_from=PATIENT,
    #     subject_to="fsaverage",
    #     spacing=None,
    #     sparse=True,
    #     subjects_dir=subjects_dir,
    # )
    # stc_fsaverage = morph.apply(stc)
    # src_fsaverage_fname = (
    #     subjects_dir + "/fsaverage/bem/fsaverage-ico-5-src.fif"
    # )
    # src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

    # plot_sparse_source_estimates(
    #     src_fsaverage,
    #     stc_fsaverage,
    #     bgcolor=(1, 1, 1),
    #     fig_name=PATIENT + "-" + ORIENTATION,
    #     opacity=0.1,
    # )
