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

import mne
from mne.viz.utils import _get_color_list


def plot_source_activations(src, stcs, axes, labels=None, linewidth=2):
    if not isinstance(stcs, list):
        stcs = [stcs]
    if labels is not None and not isinstance(labels, list):
        labels = [labels]

    colors = _get_color_list()
    linestyles = ["-", "--", ":"]

    lh_points = src[0]["rr"]

    vertnos = [
        np.r_[stc.lh_vertno, lh_points.shape[0] + stc.rh_vertno]
        for stc in stcs
    ]
    unique_vertnos = np.unique(np.concatenate(vertnos).ravel())

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    colors = cycle(colors)

    if labels is not None:
        colors = [
            next(colors)
            for _ in range(np.unique(np.concatenate(labels).ravel()).size)
        ]

    for idx, v in enumerate(unique_vertnos):
        ind = [k for k, vertno in enumerate(vertnos) if v in vertno]

        if labels is None:
            c = next(colors)
        else:
            c = colors[labels[ind[0]][vertnos[ind[0]] == v]]

        for k in ind:
            vertno = vertnos[k]
            mask = vertno == v
            assert np.sum(mask) == 1
            linestyle = linestyles[k]
            axes.plot(
                1e3 * stcs[k].times,
                1e9 * stcs[k].data[mask].ravel(),
                c=c,
                linewidth=linewidth,
                linestyle=linestyle,
            )

    # axes.set_aspect(1.1)

    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(8)

    axes.set_xlabel("Time (ms)", fontsize=8)
    axes.set_ylabel("Source amplitude (nAm)", fontsize=8)

    return fig


data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, "subjects")
fname_stc = op.join(data_path, "MEG", "sample", "sample_audvis-meg-eeg-lh.stc")
fname_evoked = op.join(data_path, "MEG", "sample", "sample_audvis-ave.fif")
fname_fwd = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"


evoked = mne.read_evokeds(fname_evoked, "Left Auditory")
evoked.pick_types(meg="grad").apply_baseline((None, 0.0))
max_t = evoked.get_peak()[1]

forward = mne.read_forward_solution(fname_fwd)
# forward = joblib.load("data/forward_lasso-sure.pkl")

# stc = mne.read_source_estimate(fname_stc)
stc = joblib.load("data/stc_adaptive-sure.pkl")

colormap = "viridis"

# Plot the STC, get the brain image, crop it:
brain = stc.plot(
    views="lat",
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
)
screenshot = brain.screenshot()
brain.close()

nonwhite_pix = (screenshot != 255).any(-1)
nonwhite_row = nonwhite_pix.any(1)
nonwhite_col = nonwhite_pix.any(0)
cropped_screenshot = screenshot[nonwhite_row]  # [:, nonwhite_col]

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

# figsize unit is inches
# fig, axes = plt.subplots(
#    nrows=2,
#    ncols=1,
#    figsize=(4.5, 3.0),
# gridspec_kw=dict(height_ratios=[4, 4]),
# )

fig = plt.figure(figsize=(4.5, 3.0))
axes = [
    plt.subplot2grid((7, 1), (0, 0), rowspan=4),
    plt.subplot2grid((7, 1), (4, 0), rowspan=3),
]

evoked_idx = 1
brain_idx = 0


plot_source_activations(forward["src"], stc, axes=axes[evoked_idx])

# now add the brain to the lower axes
axes[brain_idx].imshow(cropped_screenshot)
axes[brain_idx].axis("off")

# add a vertical colorbar with the same properties as the 3D one
divider = make_axes_locatable(axes[brain_idx])
cax = divider.append_axes("right", size="5%", pad=0.2)
cbar = mne.viz.plot_brain_colorbar(
    cax, "auto", colormap, orientation="vertical"  # label="Activation (F)",
)

# tweak margins and spacing
fig.subplots_adjust(
    left=0.15,
    right=0.9,
    bottom=0.15,
    top=0.9,
    wspace=0.1,
    hspace=0.2
    # left=0.2, right=0.9, bottom=0.15, top=0.9, wspace=0.1, hspace=0.5,
)

# add subplot labels
for ax, label in zip(axes, "AB"):
    ax.text(
        0.03,
        ax.get_position().ymax,
        label,
        transform=fig.transFigure,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )

fig.savefig("ex2.svg")

plt.show(block=True)
