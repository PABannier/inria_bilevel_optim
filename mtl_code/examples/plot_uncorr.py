import numpy as np
import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV

from utils import compute_alpha_max, plot_original_reconstructed_signal
from utils import plot_original_reconstructed_signal_band


def large_experiment_cv(X, Y, coef):
    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(5e-4, 1.8e-3, num=15)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=3)

    regressor.fit(X, Y)
    best_alpha = regressor.best_alpha_
    print("Best alpha:", best_alpha)

    coef_hat = regressor.coef_
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
        estimator = ReweightedMultiTaskLasso(best_alpha, n_iterations, False)
        estimator.fit(X, Y)

        coef_hat = estimator.coef_
        nnz_reconstructed = np.count_nonzero(
            np.count_nonzero(coef_hat, axis=1)
        )

        supports.append(nnz_reconstructed)

    fig = plt.figure(figsize=(8, 6))

    plt.plot(supports)
    plt.title(
        "Support recovery over iterations", fontweight="bold", fontsize=20
    )
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Sparsity level", fontsize=12)
    plt.show(block=True)


def plot_support_recovery_regularizing_constant(X, Y, coef):
    supports = []
    alphas = np.geomspace(1e-4, 2e-2, num=15)

    for alpha_param in alphas:
        estimator = ReweightedMultiTaskLasso(alpha_param, verbose=False)
        estimator.fit(X, Y)

        coef_hat = estimator.coef_
        nnz_reconstructed = np.count_nonzero(
            np.count_nonzero(coef_hat, axis=1)
        )

        supports.append(nnz_reconstructed)

    fig = plt.figure(figsize=(8, 6))

    xlabels = [str(round(x, 4)) for x in alphas]

    plt.plot(supports)
    plt.title(
        "Support recovery against penalizing constant",
        fontweight="bold",
        fontsize=20,
    )
    plt.xlabel("alpha", fontsize=12)
    plt.xticks(np.arange(len(alphas)), xlabels)
    plt.xticks(rotation=45)
    plt.ylabel("Sparsity level", fontsize=12)
    plt.show(block=True)


if __name__ == "__main__":
    X, Y, coef = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=2,
        corr=0,
        random_state=2020,
    )

    large_experiment_cv(X, Y, coef)

    print("\n")
    print("Plotting sparsity level against iterations...")
    plot_support_recovery_iterations(X, Y, coef)

    print("\n")
    print("Plotting sparsity level against penalizing constant...")
    plot_support_recovery_regularizing_constant(X, Y, coef)
