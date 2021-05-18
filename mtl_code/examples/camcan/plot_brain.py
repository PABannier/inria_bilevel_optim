import joblib
import os

import numpy as np
from numpy.linalg import norm

import mne
from mne.viz import plot_sparse_source_estimates


PATIENT = "CC320461"  # CC110033
ORIENTATION = "fixed"

if __name__ == "__main__":

    # Loading data
    in_file = f"stcs/sub-{PATIENT}/{ORIENTATION}.pkl"
    stc = joblib.load(in_file)

    subjects_dir = "subjects_dir"

    # Brain morphing
    morph = mne.compute_source_morph(
        stc,
        subject_from=PATIENT,
        subject_to="fsaverage",
        spacing=None,
        sparse=True,
        subjects_dir=subjects_dir,
    )
    stc_fsaverage = morph.apply(stc)
    src_fsaverage_fname = (
        subjects_dir + "/fsaverage/bem/fsaverage-ico-5-src.fif"
    )
    src_fsaverage = mne.read_source_spaces(src_fsaverage_fname)

    plot_sparse_source_estimates(
        src_fsaverage,
        stc_fsaverage,
        bgcolor=(1, 1, 1),
        fig_name=PATIENT + "-" + ORIENTATION,
        opacity=0.1,
    )
