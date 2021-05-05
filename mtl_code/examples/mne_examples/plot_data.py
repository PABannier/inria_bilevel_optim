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
parser.add_argument(
    "--condition",
    help="choice of condition. available: "
    + "Left Auditory, Right Auditory, Left visual, Right visual",
)

args = parser.parse_args()
ESTIMATOR = args.estimator
CONDITION = args.condition


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
    subjects_dir = data_path + "/subjects"

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # Handling average file
    evoked = mne.read_evokeds(
        ave_fname, condition=CONDITION, baseline=(None, 0)
    )
    evoked.crop(tmin=0.05, tmax=0.15)  # 0.05 - 0.15

    evoked = evoked.pick_types(eeg=False, meg=True)
    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return evoked, forward, noise_cov


if __name__ == "__main__":
    loose, depth = 0, 0.9  # Free orientation
    evoked, forward, noise_cov = load_data()

    folder_name = CONDITION.lower().replace(" ", "_")

    stc = joblib.load(f"{folder_name}/data/stc_{ESTIMATOR}.pkl")

    plot_sparse_source_estimates(
        forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1
    )
