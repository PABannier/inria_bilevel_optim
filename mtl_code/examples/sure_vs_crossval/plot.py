import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

import joblib

corr = 0.9

INFILE_REWEIGHTED = f"data/scores_reweighted_corr_{corr}.pkl"
INFILE_LASSO = f"data/scores_lasso_corr_{corr}.pkl"

# Read data
reweighted_scores = joblib.load(INFILE_REWEIGHTED)
lasso_scores = joblib.load(INFILE_LASSO)

alphas = reweighted_scores["alpha"]

cvs = [
    # Lasso
    {
        "mse": lasso_scores["mse"],
        "sure": lasso_scores["sure"],
    },
    # Reweighted
    {
        "mse": reweighted_scores["mse"],
        "sure": reweighted_scores["sure"],
    },
]

n_folds = lasso_scores["mse"].shape[1]

fig, axes = plt.subplots(2, 2, sharex="col", sharey="row")

for idx, cv in enumerate(cvs):
    axarr = axes[:, idx]

    for fold in range(n_folds):
        axarr[1].semilogx(alphas / alphas[0], cv["mse"][:, fold])

    axarr[0].semilogx(
        alphas / alphas[0],
        cv["sure"],
        label="mean across folds",
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

    axarr[0].set_ylabel("SURE")
    axarr[1].set_ylabel("MSE")
    axarr[1].set_xlabel(r"$\lambda / \lambda_{\mathrm{max}}$")
    axarr[0].legend()
    axarr[1].legend()

axes[0][0].set_title("Non-adaptive")
axes[0][1].set_title("Adaptive")
plt.suptitle(
    f"Comparison MSE/SURE in the adpative and non-adpative case \n Corr: {corr}",
    fontweight="bold",
)
plt.show(block=True)