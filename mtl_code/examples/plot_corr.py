import numpy as np
import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.cross_validation import MultiTaskLassoCV

from utils import compute_alpha_max, plot_original_reconstructed_signal
from utils import plot_original_reconstructed_signal_band


def experiment_cv(X, Y, coef):
    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(5e-4, 5e-3, num=20)
    regressor = MultiTaskLassoCV(alphas, n_folds=3)

    regressor.fit(X, Y)
    best_alpha = regressor.best_alpha_

    coef_hat = regressor.weights
    plot_original_reconstructed_signal_band(coef, coef_hat)

    nnz_original = np.count_nonzero(np.count_nonzero(coef, axis=1))
    nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))
    print("Number of non-zero rows in original:", nnz_original)
    print("Number of non-zero rows in reconstructed:", nnz_reconstructed)

    return best_alpha


def plot_correlation_performance(**data_params):
    best_alphas, best_cvs = list(), list()

    for rho in np.linspace(0, 0.9, 10):
        data_params["corr"] = rho
        X, Y, coef = simulate_data(**data_params)

        alphas = np.geomspace(1e-4, 5e-2, num=60)
        regressor = MultiTaskLassoCV(alphas, n_folds=3)

        regressor.fit(X, Y)
        best_alphas.append(regressor.best_alpha_)
        best_cvs.append(regressor.best_cv_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    xlabels = [str(round(x, 4)) for x in np.linspace(0, 1, 9)]

    axes[0].plot(best_alphas)
    axes[0].set_title("Regularizing constant")
    axes[0].set_xlabel("Rho", fontsize=12)
    axes[0].set_xticks(np.arange(len(best_alphas)), xlabels)
    axes[0].set_ylabel("Best alphas", fontsize=12)

    axes[1].plot(best_cvs)
    axes[1].set_title("CV MSE")
    axes[1].set_xlabel("Rho", fontsize=12)
    axes[1].set_xticks(np.arange(len(best_cvs)), xlabels)
    axes[1].set_ylabel("Best cross-validation score", fontsize=12)

    fig.suptitle("Impact on correlation coefficient on...", fontsize=13)

    plt.show(block=True)


if __name__ == "__main__":
    X, Y, coef = simulate_data(
        n_samples=50, n_features=250, n_tasks=25, nnz=25, corr=0.6, random_state=1
    )

    # experiment_cv(X, Y, coef)
    data_params = dict(
        n_samples=50, n_features=250, n_tasks=25, nnz=25, snr=3, random_state=0
    )
    plot_correlation_performance(**data_params)
