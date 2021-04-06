import argparse
import joblib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from numpy.linalg import norm

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates


parser = argparse.ArgumentParser()
parser.add_argument(
    "--estimator",
    help="choice of estimator to reconstruct the channels. available: "
    + "lasso-cv, lasso-sure, adaptive-cv, adaptive-sure",
)

args = parser.parse_args()
ESTIMATOR = args.estimator


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
    subjects_dir = data_path + "/subjects"
    condition = "Left Auditory"

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # Handling average file
    evoked = mne.read_evokeds(
        ave_fname, condition=condition, baseline=(None, 0)
    )
    evoked.crop(tmin=0.04, tmax=0.18)

    evoked = evoked.pick_types(eeg=False, meg=True)
    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov


if __name__ == "__main__":
    loose, depth = 1.0, 0  # Free orientation
    evoked, forward, noise_cov = load_data()

    stc = joblib.load(f"data/stc_{ESTIMATOR}.pkl")

    plot_sparse_source_estimates(
        forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1
    )