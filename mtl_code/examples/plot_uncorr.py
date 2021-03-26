from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMTL
from mtl.cross_validation import MultiTaskLassoCV

import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state

import matplotlib.pyplot as plt
import seaborn as sns


# ======= Utility functions =======


def compute_alpha_max(X, Y):
    B = X.T @ Y
    b = np.linalg.norm(B, axis=1)
    return np.max(b) / X.shape[0]


def plot_original_reconstructed_signal(original, reconstructed, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))

    axes[0].set_title("Original signal", fontweight="bold", fontsize=15)
    axes[0].imshow(original.T, cmap="binary")

    axes[1].set_title("Recovered signal", fontweight="bold", fontsize=15)
    axes[1].imshow(reconstructed.T, cmap="binary")

    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=18,
    )

    plt.show(block=True)


def plot_original_reconstructed_signal_band(
    original, reconstructed, title="Sparsity patterns"
):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True)

    axes[0].spy(original.T, aspect="auto")
    axes[0].xaxis.tick_bottom()
    axes[0].set_title("Original signal", fontsize=14)
    axes[0].set_ylabel("Tasks", fontsize=12)

    axes[1].spy(reconstructed.T, aspect="auto")
    axes[1].xaxis.tick_bottom()
    axes[1].set_title("Reconstructed signal", fontsize=14)

    plt.suptitle(title, fontweight="bold", fontsize=20)
    plt.ylabel("Tasks", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.show(block=True)


# ======== Experiments ============


def small_experiment_no_cv(X, Y, coef):
    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max for small experiment:", alpha_max)

    regressor = ReweightedMTL(alpha=alpha_max * 0.01)
    regressor.fit(X, Y)

    coef_hat = regressor.weights
    plot_original_reconstructed_signal(
        coef, coef_hat, "A first attempt for Reweighted Multi-Task LASSO minimization"
    )


def small_experiment_cv(X, Y, coef):
    alphas = np.geomspace(1e-3, 1e-1, num=20)
    regressor = MultiTaskLassoCV(alphas, n_folds=5)

    regressor.fit(X, Y)
    coef_hat = regressor.weights

    plot_original_reconstructed_signal(coef, coef_hat, "A cross-validated attempt...")


def large_experiment_cv(X, Y, coef):
    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(5e-4, 1.8e-3, num=15)
    regressor = MultiTaskLassoCV(alphas, n_folds=3)

    regressor.fit(X, Y)
    best_alpha = regressor.best_alpha_

    coef_hat = regressor.weights
    plot_original_reconstructed_signal_band(coef, coef_hat)

    nnz_original = np.count_nonzero(np.count_nonzero(coef, axis=1))
    nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))
    print("Number of non-zero rows in original:", nnz_original)
    print("Number of non-zero rows in reconstructed:", nnz_reconstructed)


# ======== Visualization ===========


def plot_support_recovery_iterations(X, Y, coef):
    supports = []
    best_alpha = 2e-3

    for n_iterations in range(1, 11):
        estimator = ReweightedMTL(alpha=best_alpha, verbose=False)
        estimator.fit(X, Y, n_iterations=n_iterations)

        coef_hat = estimator.weights
        nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))

        supports.append(nnz_reconstructed)

        fig = plt.figure(figsize=(8, 6))

    plt.plot(supports)
    plt.title("Support recovery over iterations", fontweight="bold", fontsize=20)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Sparsity level", fontsize=12)
    plt.show(block=True)


def plot_support_recovery_regularizing_constant(X, Y, coef):
    supports = []
    alphas = np.geomspace(1e-4, 2e-2, num=15)

    for alpha in alphas:
        estimator = ReweightedMTL(alpha=alpha, verbose=False)
        estimator.fit(X, Y)

        coef_hat = estimator.weights
        nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))

        supports.append(nnz_reconstructed)

    fig = plt.figure(figsize=(8, 6))

    xlabels = [str(round(x, 4)) for x in alphas]

    plt.plot(supports)
    plt.title(
        "Support recovery against penalizing constant", fontweight="bold", fontsize=20
    )
    plt.xlabel("alpha", fontsize=12)
    plt.xticks(np.arange(len(alphas)), xlabels)
    plt.xticks(rotation=45)
    plt.ylabel("Sparsity level", fontsize=12)
    plt.show(block=True)


if __name__ == "__main__":
    print("===== SMALL EXPERIMENT =====")
    X, Y, coef = simulate_data(
        n_samples=10, n_features=50, n_tasks=5, nnz=5, corr=0, random_state=42
    )

    small_experiment_no_cv(X, Y, coef)
    small_experiment_cv(X, Y, coef)

    print("\n")
    print("===== LARGE EXPERIMENT =====")
    X, Y, coef = simulate_data(
        n_samples=50, n_features=250, n_tasks=25, nnz=25, corr=0, random_state=2020
    )

    large_experiment_cv(X, Y, coef)

    print("\n")
    print("Plotting sparsity level against iterations...")
    plot_support_recovery_iterations(X, Y, coef)

    print("\n")
    print("Plotting sparsity level against penalizing constant...")
    plot_support_recovery_regularizing_constant(X, Y, coef)
