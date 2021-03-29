import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.linear_model import MultiTaskLassoCV

from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.simulated_data import simulate_data

from utils import compute_alpha_max


def plot_mse_path_reweighted_mtl():
    X, Y, _ = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=0.2,
        random_state=2020,
        snr=1,
    )

    n_folds = 5

    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max for large experiment:", alpha_max)
    print("\n")

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=30)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    regressor.fit(X, Y)

    plt.figure(figsize=(8, 6))

    for idx_fold in range(n_folds):
        plt.semilogx(
            alphas,
            regressor.mse_path_[:, idx_fold],
            linestyle="--",
            label=f"Fold {idx_fold + 1}",
        )

    plt.semilogx(
        alphas,
        regressor.mse_path_.mean(axis=1),
        linewidth=3,
        color="black",
        label="Mean",
    )

    plt.xlabel("alpha", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title("MSE Path - Reweighted MTL", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


def plot_comparison_mse_path_lasso():
    X, Y, _ = simulate_data(
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
    print("Alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=100)

    # Multi-task LASSO
    mtl_lasso = MultiTaskLassoCV(cv=n_folds, random_state=2020, alphas=alphas)
    mtl_lasso.fit(X, Y)

    # Reweighted multi-task LASSO
    reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl_lasso.fit(X, Y)

    plt.figure(figsize=(8, 6))

    plt.semilogx(
        alphas,
        mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="red",
        label="Non-reweighted",
    )

    # mtl_min_idx = mtl_lasso.mse_path_.mean(axis=1).argmin()
    # plt.axvline(x=alphas[mtl_min_idx], color="red")

    plt.semilogx(
        alphas,
        reweighted_mtl_lasso.mse_path_.mean(axis=1),
        linewidth=2,
        color="green",
        label="Reweighted",
    )

    # reweighted_mtl_min_idx = reweighted_mtl_lasso.mse_path_.mean(axis=1).argmin()
    # plt.axvline(x=alphas[reweighted_mtl_min_idx], color="green")

    plt.xlabel("alpha", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(
        "MSE Path - Non-reweighted vs. reweighted MTL - Corr: 0.2",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend()
    plt.show(block=True)


def plot_mse_path_reweighted_mtl_wrt_correlation():
    correlation_coefficients = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ["r", "b", "g", "y", "black"]

    plt.figure(figsize=(8, 6))

    for corr, color in zip(correlation_coefficients, colors):
        X, Y, _ = simulate_data(
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
        print("Alpha max for large experiment:", alpha_max)

        alphas = np.geomspace(alpha_max / 100, alpha_max / 5, num=30)

        reweighted_mtl_lasso = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
        reweighted_mtl_lasso.fit(X, Y)

        x = alphas
        y = reweighted_mtl_lasso.mse_path_.mean(axis=1)

        plt.semilogx(
            x, y, linewidth=1.5, linestyle="dashed", color=color, label=f"{corr}"
        )

        reweighted_mtl_min_idx = reweighted_mtl_lasso.mse_path_.mean(axis=1).argmin()
        plt.axvline(x=alphas[reweighted_mtl_min_idx], color=color)

    plt.xlabel("alpha", fontsize=12)
    plt.ylabel("MSE", fontsize=12)
    plt.title(
        "MSE Path vs. data correlation - Reweighted MTL",
        fontsize=15,
        fontweight="bold",
    )
    plt.legend()
    plt.show(block=True)


if __name__ == "__main__":
    # plot_mse_path_reweighted_mtl()
    # plot_comparison_mse_path_lasso()
    plot_mse_path_reweighted_mtl_wrt_correlation()
