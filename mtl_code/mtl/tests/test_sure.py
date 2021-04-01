import numpy as np
from numpy.linalg import norm

from mtl.sure import SURE
from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV

from examples.utils import compute_alpha_max


def test_best_alpha_sure_mse(random_state=2020):
    """Compares that the alpha that minimizes SURE and
    CV MSE are the same for a relatively small example.

    Parameters
    ----------
    random_state: int or None, default=2020
        For reproducibility purposes, the seed is set.
    """
    X, Y, _, sigma = simulate_data(
        n_samples=50,
        n_features=250,
        n_tasks=25,
        nnz=10,
        corr=0.2,
        random_state=random_state,
        snr=2,
    )

    n_folds = 5

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    alpha_max = compute_alpha_max(X, Y)

    alphas = np.geomspace(alpha_max / 100, alpha_max, num=20)
    sure_metrics = []
    mse_metrics = None

    reweighted_mtl = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    reweighted_mtl.fit(X, Y)

    regressor = SURE(sigma, random_state=random_state)

    for alpha in alphas:
        val = regressor.get_val(X, Y, alpha)
        sure_metrics.append(val)

    sure_metrics = np.array(sure_metrics)

    min_idx = sure_metrics.argmin()
    min_idx_2 = reweighted_mtl.mse_path_.mean(axis=1).argmin()

    assert alphas[min_idx] == alphas[min_idx_2]
