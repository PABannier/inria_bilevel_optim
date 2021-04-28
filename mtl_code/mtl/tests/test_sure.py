import numpy as np
from numpy.linalg import norm

from mtl.sure import SURE
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.mtl import ReweightedMultiTaskLasso

from mtl.utils_datasets import compute_alpha_max


if __name__ == "__main__":
    random_state = 2020

    X, Y, _, sigma = simulate_data(
        n_samples=15,
        n_features=20,
        n_tasks=5,
        nnz=2,
        random_state=random_state,
        snr=3,
    )

    alpha_max = compute_alpha_max(X, Y)
    alphas = np.geomspace(alpha_max, alpha_max / 100, num=20)

    criterion = SUREForReweightedMultiTaskLasso(sigma, alphas, random_state)

    best_sure, best_alpha = criterion.get_val(X, Y)

    criterion2 = SURE(
        ReweightedMultiTaskLasso,
        sigma,
        random_state=random_state,
    )

    sure_metrics = []

    for alpha in alphas:
        val = criterion2.get_val(X, Y, alpha, verbose=False)
        sure_metrics.append(val)

    sure_metrics = np.array(sure_metrics)

    min_idx = sure_metrics.argmin()

    assert best_alpha == alphas[min_idx]


# def test_warm_start(random_state=2020):


# def test_best_alpha_sure_mse(random_state=2020):
#     """Compares that the alpha that minimizes SURE and
#     CV MSE are the same for a relatively small example.

#     Parameters
#     ----------
#     random_state: int or None, default=2020
#         For reproducibility purposes, the seed is set.
#     """
#     X, Y, _, sigma = simulate_data(
#         n_samples=15,
#         n_features=20,
#         n_tasks=5,
#         nnz=2,
#         random_state=random_state,
#         snr=3,
#     )

#     n_folds = 5

#     X = np.asfortranarray(X)
#     Y = np.asfortranarray(Y)

#     alpha_max = compute_alpha_max(X, Y)

#     alphas = np.geomspace(alpha_max, alpha_max * 0.01, num=20)
#     sure_metrics = []
#     mse_metrics = None

#     reweighted_mtl = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
#     reweighted_mtl.fit(X, Y)

#     criterion = SURE(
#         ReweightedMultiTaskLasso, sigma, random_state=random_state
#     )

#     for alpha in alphas:
#         val = criterion.get_val(X, Y, alpha)
#         sure_metrics.append(val)

#     sure_metrics = np.array(sure_metrics)

#     min_idx = sure_metrics.argmin()
#     min_idx_2 = reweighted_mtl.mse_path_.mean(axis=1).argmin()

#     assert alphas[min_idx] == alphas[min_idx_2]
