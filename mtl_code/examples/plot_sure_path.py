import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from mtl.sure import SURE
from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV

from utils import compute_alpha_max, plot_sure_mse_path


def sure_mse_path(snr=2, corr=0.5, random_state=2020):
    X, Y, _, sigma = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=corr,
        random_state=random_state,
        snr=snr,
    )

    n_folds = 5

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=50)
    sure_metrics = []
    mse_metrics = None

    reweighted_mtl = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl.fit(X, Y)

    regressor = SURE(sigma, random_state=random_state)

    for alpha in alphas:
        print("Computing SURE for alpha =", alpha)
        val = regressor.get_val(X, Y, alpha)
        sure_metrics.append(val)

    sure_metrics = np.array(sure_metrics)
    sure_metrics /= sure_metrics.mean()

    mse_metrics = reweighted_mtl.mse_path_.mean(axis=1)
    mse_metrics /= mse_metrics.mean()

    plot_sure_mse_path(
        alphas, alpha_max, mse_metrics, sure_metrics, reweighted_mtl.mse_path_
    )


def impact_correlation_coefficient_on_sure(random_state=2020):
    correlation_coefficients = [0.1, 0.3, 0.5, 0.7, 0.9]

    sure_metrics = []
    mse_metrics = []

    for corr in correlation_coefficients:
        print("Computing SURE for rho =", corr)

        X, Y, _, sigma = simulate_data(
            n_samples=50,
            n_features=250,
            n_tasks=25,
            nnz=8,
            corr=corr,
            random_state=random_state,
            snr=0.8,
        )

        X = np.asfortranarray(X)
        Y = np.asfortranarray(Y)

        alpha_max = compute_alpha_max(X, Y)
        print("alpha max for large experiment:", alpha_max)

        alphas = np.geomspace(alpha_max / 100, alpha_max / 5, num=30)

        n_folds = 5

        regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
        regressor.fit(X, Y)
        best_alpha = regressor.best_alpha_

        sure_approx = SURE(sigma, random_state=random_state)
        sure_metrics.append(sure_approx.get_val(X, Y, best_alpha))

        mse_metrics.append(regressor.best_cv_)

    sure_metrics = np.array(sure_metrics)
    mse_metrics = np.array(mse_metrics)

    sure_metrics /= sure_metrics.mean()
    mse_metrics /= mse_metrics.mean()

    plt.figure(figsize=(8, 6))

    plt.semilogx(
        correlation_coefficients,
        mse_metrics,
        linestyle="--",
        label="MSE",
        color="midnightblue",
    )
    plt.semilogx(
        correlation_coefficients,
        sure_metrics,
        linestyle="--",
        label="SURE",
        color="orange",
    )

    plt.xlabel("Correlation coefficient", fontsize=12)
    plt.ylabel("MSE / SURE (normalized)", fontsize=12)
    plt.title("Impact of rho on MSE and SURE", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    sure_mse_path()
    impact_correlation_coefficient_on_sure()
