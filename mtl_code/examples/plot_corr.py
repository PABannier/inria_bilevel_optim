from mtl.simulated_data import simulate_data
from mtl.cross_validation import MultiTaskLassoCV
from utils import compute_alpha_max, plot_original_reconstructed_signal
from utils import plot_original_reconstructed_signal_band
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    X, Y, coef = simulate_data(
        n_samples=50, n_features=250, n_tasks=25, nnz=25, corr=0.4, random_state=0
    )

    experiment_cv(X, Y, coef)
