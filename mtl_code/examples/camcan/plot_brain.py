import joblib
import os
import glob

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

import mne
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.mxne_inverse import _compute_residual

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


def merge_brain_plot(subject_ids, fixed=True):
    orient = "Fixed" if fixed else "Free"

    subjects_dir = "subjects_dir"
    src_fsaverage_fname = (
        subjects_dir + "/fsaverage/bem/fsaverage-ico-5-src.fif"
    )
    src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

    stcs = []

    for subject_id in subject_ids:
        if fixed:
            path = f"stcs/sub-{subject_id}/free.pkl"
        else:
            path = f"stcs/sub-{subject_id}/fixed.pkl"

        stc = joblib.load(path)

        morph = mne.compute_source_morph(
            stc,
            subject_from=subject_id,
            subject_to="fsaverage",
            spacing=None,
            sparse=True,
            subjects_dir=subjects_dir,
        )
        stc_fsaverage = morph.apply(stc)
        stcs.append(stc_fsaverage)

    plot_sparse_source_estimates(
        src_fsaverage,
        stcs,
        bgcolor=(1, 1, 1),
        fig_name=f"Merged - {orient}",
        opacity=0.1,
        plot_merged_sources=True,
    )


if __name__ == "__main__":
    # merge_brain_plot(WORKING_EXAMPLES[:4])

    # Loading data
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
