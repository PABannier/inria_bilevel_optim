import argparse

import joblib
from itertools import cycle

import os.path as op

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import (
    make_axes_locatable,
    ImageGrid,
    inset_locator,
)

from numba import njit

import mne
from mne.viz.utils import _get_color_list

plt.rcParams.update(
    {
        "ytick.labelsize": "small",
        "xtick.labelsize": "small",
        "axes.labelsize": "small",
        "axes.titlesize": "medium",
        "grid.color": "0.75",
        "grid.linestyle": ":",
    }
)

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

if args.estimator is None or args.condition is None:
    raise ValueError("You need to choose an estimator and a condition.")
else:
    ESTIMATOR = args.estimator
    CONDITION = args.condition


def add_foci_to_brain_surface(brain, stc, ax):
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


data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, "subjects")
fname_evoked = op.join(data_path, "MEG", "sample", "sample_audvis-ave.fif")
fname_fwd = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"


evoked = mne.read_evokeds(fname_evoked, "Left Auditory")
evoked.pick_types(meg="grad").apply_baseline((None, 0.0))
max_t = evoked.get_peak()[1]

forward = mne.read_forward_solution(fname_fwd)

folder_name = CONDITION.lower().replace(" ", "_")
stc = joblib.load(f"{folder_name}/data/stc_{ESTIMATOR}.pkl")

lower_bound = round(stc.data.min() * 1e9)
upper_bound = round(stc.data.max() * 1e9)


colormap = "inferno"
clim = dict(
    kind="value",
    lims=(lower_bound, 0, upper_bound),
)


# Plot the STC, get the brain image, crop it:
brain = stc.plot(
    views=["lat", "med"],
    hemi="split",
    size=(1000, 500),
    subject="sample",
    subjects_dir=subjects_dir,
    initial_time=max_t,
    background="w",
    clim="auto",
    colorbar=False,
    colormap=colormap,
    time_viewer=False,
    show_traces=False,
    cortex="bone",
    volume_options=dict(resolution=1),
)

brain.set_time(0.05)


fig = plt.figure(figsize=(4.5, 4.5))
axes = [
    plt.subplot2grid((7, 1), (0, 0), rowspan=4),
    plt.subplot2grid((7, 1), (4, 0), rowspan=3),
]

add_foci_to_brain_surface(brain, stc, axes[1])

screenshot = brain.screenshot()
brain.close()

nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)


@njit
def add_margin(nonwhite_col, margin=5):
    margin_nonwhite_col = nonwhite_col.copy()
    for i in range(len(nonwhite_col)):
        if nonwhite_col[i] == True and nonwhite_col[i - 1] == False:
            margin_nonwhite_col[i - (1 + margin) : i - 1] = True
        elif nonwhite_col[i] == False and nonwhite_col[i - 1] == True:
            margin_nonwhite_col[i : i + margin] = True
    return margin_nonwhite_col


# Add blank columns for margin
nonwhite_col = add_margin(nonwhite_col)

cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]

evoked_idx = 1
brain_idx = 0

axes[brain_idx].imshow(cropped_screenshot)
axes[brain_idx].axis("off")

# tweak margins and spacing
fig.subplots_adjust(
    left=0.15, right=0.9, bottom=0.15, top=0.9, wspace=0.1, hspace=0.2
)

fig_dir = (
    "../../../tex/article/srcimages/blobs/"
    + f"blob-{folder_name}-{ESTIMATOR}.svg"
)

fig_dir_2 = (
    "../../../tex/article/srcimages/blobs/"
    + f"blob-{folder_name}-{ESTIMATOR}.pdf"
)

fig.savefig(fig_dir)
fig.savefig(fig_dir_2)

# plt.show()
