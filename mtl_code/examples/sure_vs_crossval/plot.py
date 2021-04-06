import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corr", help="Correlation coefficient of the design matrix"
)
args = parser.parse_args()

CORR = args.corr

INFILE_REWEIGHTED = f"data/scores_reweighted_corr_{CORR}.pkl"
INFILE_LASSO = f"data/scores_lasso_corr_{CORR}.pkl"

# Read data
reweighted_scores = joblib.load(INFILE_REWEIGHTED)
lasso_scores = joblib.load(INFILE_LASSO)

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

n_folds = lasso_scores["mse"].shape[1]

fig, axes = plt.subplots(3, 2, figsize=(8, 6), sharex="col", sharey="row")

for idx, cv in enumerate(cvs):
    axarr = axes[:, idx]

    for fold in range(n_folds):
        axarr[1].semilogx(alphas / alphas[0], cv["mse"][:, fold])
        axarr[2].semilogx(alphas / alphas[0], cv["f1"][:, fold])

    axarr[0].semilogx(
        alphas / alphas[0],
        cv["sure"],
        color="k",
        lw=2,
    )
    axarr[1].semilogx(
        alphas / alphas[0],
        cv["mse"].mean(axis=1),
        label="mean across folds",
        color="k",
        lw=2,
    )
    axarr[2].semilogx(
        alphas / alphas[0],
        cv["f1"].mean(axis=1),
        label="mean across folds",
        color="k",
        lw=2,
    )

    axarr[0].axvline(
        alphas[np.array(cv["sure"]).argmin()] / alphas[0],
        linestyle="--",
        lw=3,
        color="k",
        label=r"best $\lambda$",
    )
    axarr[1].axvline(
        alphas[np.argmin(cv["mse"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=3,
        color="k",
        label=r"best $\lambda$",
    )
    axarr[2].axvline(
        alphas[np.argmax(cv["f1"].mean(axis=1))] / alphas[0],
        linestyle="--",
        lw=3,
        color="k",
        label=r"best $\lambda$",
    )

    axarr[0].set_ylabel("SURE")
    axarr[1].set_ylabel("MSE")
    axarr[2].set_ylabel("F1")
    axarr[2].set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
    axarr[0].legend()
    axarr[1].legend()
    axarr[2].legend()

axes[0][0].set_title("Non-adaptive")
axes[0][1].set_title("Adaptive")
plt.suptitle(f"Corr: {CORR}")
plt.show(block=True)

# Save figure in tex article folder

DESTINATION_PATH = f"../../../tex/article/srcimages/sure_vs_mse_corr_{int(float(CORR)*100)}.png"

fig.savefig(DESTINATION_PATH)
print("Figure saved.")