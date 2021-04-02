import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from sklearn.linear_model import MultiTaskLassoCV

from mtl.sure import SURE
from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMultiTaskLasso
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


def plot_support_recovery(random_state=0):
    # Generating data
    corr = 0
    n_iterations = 5

    X, Y, coef, sigma = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=2,
        corr=corr,
        random_state=random_state,
        snr=0.8,
    )

    n_folds = 5
    colors = pl.cm.jet(np.linspace(0, 1, n_folds))

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 10, alpha_max, num=50)

    # Modelling and fitting data
    reweigthed_cv = ReweightedMultiTaskLassoCV(
        alphas, n_folds=n_folds, n_iterations=n_iterations
    )
    reweigthed_cv.fit(X, Y, coef_true=coef)

    sure_estimator = SURE(sigma, random_state=random_state)
    sure_metrics = []

    for alpha in alphas:
        sure_val = sure_estimator.get_val(X, Y, alpha)
        sure_metrics.append(sure_val)

    # Displaying
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex="col")

    titles = ["MSE", "SURE", "F1", "Jaccard"]

    for idx, title in enumerate(titles):
        i = idx // 2
        j = idx % 2
        axes[i][j].set_title(titles[idx])
        # axes[i][j].set_ylabel(titles[idx])
        axes[i][j].set_xlabel("$\lambda / \lambda_{\max}$")

        if title == "F1":
            path = reweigthed_cv.f1_path_
            vline_idx = reweigthed_cv.f1_path_.mean(axis=1).argmax()
        elif title == "Jaccard":
            path = reweigthed_cv.jaccard_path_
            vline_idx = reweigthed_cv.jaccard_path_.mean(axis=1).argmax()
        elif title == "MSE":
            path = reweigthed_cv.mse_path_
            vline_idx = reweigthed_cv.mse_path_.mean(axis=1).argmin()
        elif title == "SURE":
            vline_idx = np.array(sure_metrics).argmin()

        if title != "SURE":
            for idx_fold in range(n_folds):
                axes[i][j].semilogx(
                    alphas / alpha_max,
                    path[:, idx_fold],
                    linestyle="--",
                    color=colors[idx_fold],
                )

            axes[i][j].semilogx(
                alphas / alpha_max,
                path.mean(axis=1),
                linewidth=3,
                color="black",
                label="Mean",
            )

            axes[i][j].axvline(
                x=alphas[vline_idx] / alpha_max,
                color="midnightblue",
                linestyle="dashed",
                linewidth=3,
                label="Best $\lambda$",
            )

        else:
            axes[i][j].semilogx(
                alphas / alpha_max, sure_metrics, color="deepskyblue"
            )

            axes[i][j].axvline(
                x=alphas[vline_idx] / alpha_max,
                color="midnightblue",
                linestyle="dashed",
                linewidth=3,
                label="Best $\lambda$",
            )

        axes[i][j].legend()

    plt.show()


if __name__ == "__main__":
    # sure_mse_path()
    # impact_correlation_coefficient_on_sure()
    plot_support_recovery()
