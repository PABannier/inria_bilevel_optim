import joblib

import numpy as np

from surfer import Brain
from mayavi import mlab

from mne.datasets import sample
from mne.inverse_sparse.mxne_inverse import _make_sparse_stc

fig_dir = "../../../tex/article/srcimages/blobs/"


def plot_blob(
    stc,
    subject="sample",
    surface="white",
    s=18,
    save_fname="",
    data_path=sample.data_path(),
    subject_name="/subjects",
    fig_dir="",
    figsize=(800, 800),
    event_id=1,
):

    subjects_dir = data_path + subject_name
    list_hemi = ["lh", "rh"]

    for i, hemi in enumerate(list_hemi):
        figure = mlab.figure(size=figsize)
        brain = Brain(
            subject,
            hemi,
            surface,
            subjects_dir=subjects_dir,
            offscreen=False,
            figure=figure,
        )
        surf = brain.geo[hemi]
        sources_h = stc.vertices[i]  # 0 for lh, 1 for rh
        for sources in sources_h:
            mlab.points3d(
                surf.x[sources],
                surf.y[sources],
                surf.z[sources],
                color=(1, 0, 0),
                scale_factor=s,
                opacity=1.0,
                transparent=True,
            )
        if save_fname:
            fname = fig_dir + hemi + save_fname
            if event_id == 1 or event_id == 2:
                brain.save_montage(fname, order=["lat"])
            else:
                brain.save_montage(fname, order=["lat"])
                # brain.save_montage(fname, order=['ven'])

            # mlab.savefig(fname)
            figure = mlab.gcf()
            mlab.close(figure)


if __name__ == "__main__":
    estimator = "lasso-sure"
    stc = joblib.load(f"data/stc_{estimator}.pkl")

    plot_blob(
        stc, fig_dir=fig_dir, save_fname=f"-blob-left-auditory-{estimator}.png"
    )
