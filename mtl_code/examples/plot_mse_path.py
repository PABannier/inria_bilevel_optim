import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from sklearn.linear_model import MultiTaskLassoCV

from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.simulated_data import simulate_data

from utils import compute_alpha_max


def plot_mse_path_reweighted_mtl():
    X, Y, _, _ = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=0.2,
        random_state=2020,
        snr=1,
    )

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    n_folds = 5

    colors = pl.cm.jet(np.linspace(0, 1, n_folds))

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment:", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 10, alpha_max, num=20)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    regressor.fit(X, Y)

    plt.figure(figsize=(8, 6))

    for idx_fold in range(n_folds):
        plt.semilogx(
            alphas / alpha_max,
            regressor.mse_path_[:, idx_fold],
            linestyle="--",
            color=colors[idx_fold],
            label=f"Fold {idx_fold + 1}",
        )

    plt.semilogx(
        alphas / alpha_max,
        regressor.mse_path_.mean(axis=1),
        linewidth=3,
        color="black",
        label="Mean",
    )

    mtl_min_idx = regressor.mse_path_.mean(axis=1).argmin()
    plt.axvline(
        x=alphas[mtl_min_idx] / alpha_max,
        color="black",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$",
    )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("MSE path - Reweighted MTL", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


def plot_comparison_mse_path_lasso():
    X, Y, _, _ = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=0.2,
        random_state=2020,
        snr=1,
    )

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    n_folds = 5

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(alpha_max / 10, alpha_max, num=50)

    # Multi-task LASSO
    mtl_lasso = MultiTaskLassoCV(cv=n_folds, random_state=2020, alphas=alphas)
    mtl_lasso.fit(X, Y)

    # Reweighted multi-task LASSO
    reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl_lasso.fit(X, Y)

    fig = plt.figure(figsize=(8, 6))

    plt.semilogx(
        alphas / alpha_max,
        mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="deepskyblue",
        label="Non-reweighted",
    )

    mtl_min_idx = mtl_lasso.mse_path_.mean(axis=1).argmin()
    plt.axvline(
        x=alphas[mtl_min_idx] / alpha_max,
        color="deepskyblue",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$",
    )

    plt.semilogx(
        alphas / alpha_max,
        reweighted_mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="midnightblue",
        label="Reweighted",
    )

    mtl_min_idx = reweighted_mtl_lasso.mse_path_.mean(axis=1).argmin()
    plt.axvline(
        x=alphas[mtl_min_idx] / alpha_max,
        color="midnightblue",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$ - Reweighted",
    )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(
        "MSE path - Non-reweighted vs. reweighted MTL - Corr: 0.2",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend()
    plt.show(block=True)


def plot_comparison_mse_path_lasso_across_folds():
    X, Y, _, _ = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=0.2,
        random_state=2020,
        snr=1,
    )

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    n_folds = 5

    colors = pl.cm.jet(np.linspace(0, 1, n_folds))

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(alpha_max / 10, alpha_max, num=30)

    # Multi-task LASSO
    mtl_lasso = MultiTaskLassoCV(cv=n_folds, random_state=2020, alphas=alphas)
    mtl_lasso.fit(X, Y)

    # Reweighted multi-task LASSO
    reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl_lasso.fit(X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey="row")

    for idx_fold in range(n_folds):
        axes[0].semilogx(
            alphas / alpha_max,
            mtl_lasso.mse_path_[:, idx_fold],
            linestyle="--",
            color=colors[idx_fold],
            label=f"Fold {idx_fold + 1}",
        )

    axes[0].semilogx(
        alphas / alpha_max,
        mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="black",
        label="Mean",
    )

    mtl_min_idx = mtl_lasso.mse_path_.mean(axis=1).argmin()
    axes[0].axvline(
        x=alphas[mtl_min_idx] / alpha_max,
        color="black",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$",
    )

    for idx_fold in range(n_folds):
        axes[1].semilogx(
            alphas / alpha_max,
            reweighted_mtl_lasso.mse_path_[:, idx_fold],
            linestyle="--",
            color=colors[idx_fold],
            label=f"Fold {idx_fold + 1}",
        )

    axes[1].semilogx(
        alphas / alpha_max,
        reweighted_mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="black",
        label="Mean",
    )

    reweighted_mtl_min_idx = reweighted_mtl_lasso.mse_path_.mean(axis=1).argmin()
    axes[1].axvline(
        x=alphas[reweighted_mtl_min_idx] / alpha_max,
        color="black",
        linestyle="dashed",
        linewidth=3,
        label="Best $\lambda$",
    )

    axes[0].set_xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    axes[1].set_xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    axes[0].set_ylabel("MSE", fontsize=12)

    axes[0].set_title(
        "Non-reweighted MTL",
        fontsize=15,
        fontweight="bold",
    )

    axes[1].set_title(
        "Reweighted MTL",
        fontsize=15,
        fontweight="bold",
    )

    fig.suptitle("MSE path", fontsize=20, fontweight="bold")

    axes[0].legend(loc=1)
    axes[1].legend(loc=1)
    plt.show(block=True)


def plot_mse_path_reweighted_mtl_wrt_correlation():
    correlation_coefficients = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = pl.cm.jet(np.linspace(0, 1, len(correlation_coefficients)))

    plt.figure(figsize=(8, 6))

    for corr, color in zip(correlation_coefficients, colors):
        X, Y, _, _ = simulate_data(
            n_samples=50,
            n_features=250,
            n_tasks=25,
            nnz=8,  # 10 (try 8, 5)
            corr=corr,
            random_state=2020,
            snr=0.8,
        )

        X = np.asfortranarray(X)
        Y = np.asfortranarray(Y)

        n_folds = 5

        alpha_max = compute_alpha_max(X, Y)
        print("alpha max for large experiment:", alpha_max)

        alphas = np.geomspace(alpha_max / 10, alpha_max, num=30)

        reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
        reweighted_mtl_lasso.fit(X, Y)

        x = alphas
        y = reweighted_mtl_lasso.mse_path_.mean(axis=1)

        plt.semilogx(
            x / alpha_max,
            y,
            linewidth=1.5,
            linestyle="dashed",
            color=color,
            label=f"{corr}",
        )

        reweighted_mtl_min_idx = reweighted_mtl_lasso.mse_path_.mean(axis=1).argmin()
        plt.axvline(x=alphas[reweighted_mtl_min_idx] / alpha_max, color=color)

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(
        "MSE path vs. data correlation - Reweighted MTL",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend()
    plt.show(block=True)


def plot_mse_path_wrt_num_iterations(corr=0.2):
    n_iterations = [1, 3, 5, 7]
    colors = pl.cm.jet(np.linspace(0, 1, len(n_iterations)))

    dict_alpha_y = {}

    X, Y, _, _ = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=8,
        corr=corr,
        random_state=2020,
        snr=1,
    )

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    n_folds = 5

    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(alpha_max / 50, alpha_max, num=60)

    for n_iter, color in zip(n_iterations, colors):
        reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
        reweighted_mtl_lasso.fit(X, Y, n_iterations=n_iter)

        y = reweighted_mtl_lasso.mse_path_.mean(axis=1)

        dict_alpha_y[n_iter] = alphas, y

    plt.figure(figsize=(8, 6))
    for color, n_iter in zip(colors, dict_alpha_y.keys()):
        alpha_param, y = dict_alpha_y[n_iter]
        plt.semilogx(
            alpha_param / alpha_max, y, color=color, linewidth=1.5, label=f"{n_iter}"
        )

        reweighted_mtl_min_idx = y.argmin()
        plt.axvline(
            x=alphas[reweighted_mtl_min_idx] / alpha_max,
            color=color,
            linewidth=3,
            linestyle="--",
        )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(
        f"MSE path vs. number of iterations - Reweighted MTL - Corr: {corr}",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    plot_mse_path_reweighted_mtl()
    plot_comparison_mse_path_lasso()
    plot_comparison_mse_path_lasso_across_folds()
    plot_mse_path_reweighted_mtl_wrt_correlation()
    plot_mse_path_wrt_num_iterations(corr=0.5)
