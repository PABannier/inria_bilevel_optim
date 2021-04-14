from os import path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.forward import make_forward_dipole
from mne.evoked import combine_evoked
from mne.simulation import simulate_evoked

from nilearn.plotting import plot_anat
from nilearn.datasets import load_mni152_template

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, "subjects")
fname_ave = op.join(data_path, "MEG", "sample", "sample_audvis-ave.fif")
fname_cov = op.join(data_path, "MEG", "sample", "sample_audvis-cov.fif")
fname_bem = op.join(subjects_dir, "sample", "bem", "sample-5120-bem-sol.fif")
fname_trans = op.join(
    data_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
)
fname_surf_lh = op.join(subjects_dir, "sample", "surf", "lh.white")

evoked = mne.read_evokeds(
    fname_ave, condition="Right visual", baseline=(None, 0)
)
evoked.pick_types(meg=True, eeg=False)
# evoked_full = evoked.copy()
# evoked.crop(0.07, 0.08)

# Fit a dipole
# dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
# dip.plot_locations(fname_trans, "sample", subjects_dir, mode="orthoview")

evoked.plot()
