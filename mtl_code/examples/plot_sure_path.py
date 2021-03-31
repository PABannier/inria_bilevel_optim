import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from mtl.sure import SURE
from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV

from utils import compute_alpha_max


def plot_sure_path(corr=0.2, random_state=2020):
    X, Y, coef = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=corr,
        random_state=random_state,
        snr=1,
    )

    sigma = 0.025  # ????

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=20)
    sure_metrics = []

    regressor = SURE(sigma, random_state=random_state)

    for alpha in alphas:
        print("Computing SURE for alpha=", alpha)
        val = regressor.get_val(X, Y, alpha)
        sure_metrics.append(val)

    plt.figure(figsize=(8, 6))

    plt.semilogx(alphas / alpha_max, sure_metrics, linestyle="--", label="SURE")

    min_idx = np.array(sure_metrics).argmin()
    plt.axvline(
        x=alphas[min_idx] / alpha_max,
        color="black",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$",
    )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("SURE", fontsize=12)
    plt.title("SURE path - Reweighted MTL", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


def plot_sure_mse_path(corr=0.2, random_state=2020):
    X, Y, coef = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=corr,
        random_state=random_state,
        snr=1,
    )

    sigma = 0.025  # ????
    n_folds = 5

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=20)
    sure_metrics = []
    mse_metrics = None

    reweighted_mtl = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl.fit(X, Y)

    regressor = SURE(sigma, random_state=random_state)

    for alpha in alphas:
        print("Computing SURE for alpha=", alpha)
        val = regressor.get_val(X, Y, alpha)
        sure_metrics.append(val)

    sure_metrics = np.array(sure_metrics)
    sure_metrics /= sure_metrics.mean()

    mse_metrics = reweighted_mtl.mse_path_.mean(axis=1)
    mse_metrics /= mse_metrics.mean()

    plt.figure(figsize=(8, 6))

    plt.semilogx(
        alphas / alpha_max,
        mse_metrics,
        linestyle="--",
        label="MSE",
        color="midnightblue",
    )
    plt.semilogx(
        alphas / alpha_max,
        sure_metrics,
        linestyle="--",
        label="SURE",
        color="orange",
    )

    min_idx = sure_metrics.argmin()
    plt.axvline(
        x=alphas[min_idx] / alpha_max,
        color="orange",
        linestyle="dashed",
        linewidth=3,
        label="Best SURE $\lambda$",
    )

    min_idx_2 = reweighted_mtl.mse_path_.mean(axis=1).argmin()
    plt.axvline(
        x=alphas[min_idx] / alpha_max,
        color="midnightblue",
        linestyle="dashed",
        linewidth=3,
        label="Best MSE $\lambda$",
    )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE / SURE (normalized)", fontsize=12)
    plt.title("MSE vs SURE paths", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    plot_sure_mse_path()
