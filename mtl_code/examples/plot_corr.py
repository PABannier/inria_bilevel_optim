import numpy as np
import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV

from mtl.utils_datasets import (
    compute_alpha_max,
    plot_original_reconstructed_signal,
    plot_original_reconstructed_signal_band,
)


def experiment_cv(X, Y, coef):
    alpha_max = compute_alpha_max(X, Y)
    print("alpha max for large experiment:", alpha_max)

    alphas = np.geomspace(alpha_max * 0.2, alpha_max * 0.08, num=20)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=3)

    regressor.fit(X, Y)
    best_alpha = regressor.best_alpha_

    coef_hat = regressor.coef_
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
        X, Y, coef, sigma = simulate_data(**data_params)

        alphas = np.geomspace(5e-2, 1e-4, num=15)
        regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=3)

        regressor.fit(X, Y)
        best_alphas.append(regressor.best_alpha_)
        best_cvs.append(regressor.best_cv_)

    xlabels = [str(round(x, 2)) for x in np.linspace(0, 0.9, 10)]

    fig = plt.figure(figsize=(6, 4))

    plt.plot(best_cvs)
    plt.xlabel("Rho", fontsize=12)
    plt.xticks(np.arange(len(best_cvs)), xlabels)
    plt.ylabel("Best cross-validation score", fontsize=12)
    plt.title("Impact of correlation coefficient on CV MSE", fontsize=13)

    plt.show(block=True)


if __name__ == "__main__":
    print("====== Running experiment CV ======")
    X, Y, coef, _ = simulate_data(
        n_samples=10,
        n_features=15,
        n_tasks=13,
        nnz=3,
        snr=3,
        corr=0.6,
        random_state=1,
    )
    experiment_cv(X, Y, coef)

    print("====== Plotting rho vs MSE ======")
    data_params = dict(
        n_samples=10, n_features=15, n_tasks=13, nnz=3, snr=3, random_state=0
    )
    plot_correlation_performance(**data_params)
