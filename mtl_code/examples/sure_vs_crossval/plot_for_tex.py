import joblib

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

corrs = [0.3, 0.7, 0.99]


def load_data_for_corr(corr):
    infile_reweighted = f"data/scores_reweighted_corr_{corr}.pkl"
    infile_lasso = f"data/scores_lasso_corr_{corr}.pkl"

    reweighted_scores = joblib.load(infile_reweighted)
    lasso_scores = joblib.load(infile_lasso)

    alphas = reweighted_scores["alpha"]

    cvs = [
        {
            "f1": lasso_scores["f1"],
            "mse": lasso_scores["mse"],
            "sure": lasso_scores["sure"],
        },
        {
            "f1": reweighted_scores["f1"],
            "mse": reweighted_scores["mse"],
            "sure": reweighted_scores["sure"],
        },
    ]

    return alphas, cvs


fig, axes = plt.subplots(3, len(corrs), sharex="col", sharey="row")

for idx, corr in enumerate(corrs):
    axarr = axes[:, idx]
    alphas, cvs = load_data_for_corr(corr)

    # MSE
    axarr[0].semilogx(
        alphas / alphas[0],
        cvs[0]["mse"].mean(axis=1),
        color="lightskyblue",
        lw=1,
        label="lasso",
    )
    axarr[0].semilogx(
        alphas / alphas[0],
        cvs[1]["mse"].mean(axis=1),
        color="slateblue",
        lw=1,
        label="reweighted",
    )
    axarr[0].axvline(
        alphas[np.argmin(cvs[0]["mse"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=2,
        color="lightskyblue",
        label=r"lasso - best $\lambda$",
    )
    axarr[0].axvline(
        alphas[np.argmin(cvs[1]["mse"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=2,
        color="slateblue",
        label=r"reweighted - best $\lambda$",
    )

    # SURE
    axarr[1].semilogx(
        alphas / alphas[0],
        cvs[0]["sure"],
        color="lightskyblue",
        lw=1,
        label="lasso",
    )
    axarr[1].semilogx(
        alphas / alphas[0],
        cvs[1]["sure"],
        color="slateblue",
        lw=1,
        label="reweighted",
    )
    axarr[1].axvline(
        alphas[np.array(cvs[0]["sure"]).argmin()] / alphas[0],
        linestyle="--",
        lw=2,
        color="lightskyblue",
        label=r"lasso - best $\lambda$",
    )
    axarr[1].axvline(
        alphas[np.array(cvs[1]["sure"]).argmin()] / alphas[0],
        linestyle="--",
        lw=2,
        color="slateblue",
        label=r"reweighted - best $\lambda$",
    )

    # F1
    axarr[2].semilogx(
        alphas / alphas[0],
        cvs[0]["f1"].mean(axis=1),
        color="lightskyblue",
        lw=1,
        label="lasso",
    )
    axarr[2].semilogx(
        alphas / alphas[0],
        cvs[1]["f1"].mean(axis=1),
        color="slateblue",
        lw=1,
        label="reweighted",
    )
    axarr[2].axvline(
        alphas[np.argmax(cvs[0]["f1"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=2,
        color="lightskyblue",
        label=r"lasso - best $\lambda$",
    )
    axarr[2].axvline(
        alphas[np.argmax(cvs[1]["f1"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=2,
        color="slateblue",
        label=r"reweighted - best $\lambda$",
    )

    axarr[2].set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
    axarr[0].set_title(f"Corr = {corr}", fontweight=600)

axes[0][0].set_ylabel("MSE")
axes[1][0].set_ylabel("SURE")
axes[2][0].set_ylabel("F1")


handles, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(right=0.65)
plt.show(block=True)

OUT_PATH_1 = f"../../../tex/article/srcimages/sure_mse_f1_comparison.svg"
OUT_PATH_2 = f"../../../tex/article/srcimages/sure_mse_f1_comparison.pdf"

fig.savefig(OUT_PATH_1)
fig.savefig(OUT_PATH_2)
print("Figure saved.")
