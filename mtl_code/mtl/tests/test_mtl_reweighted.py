import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state

from mtl.mtl import ReweightedMTL
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.simulated_data import simulate_data


def test_training_loss_decrease():
    X, Y, W, _ = simulate_data(n_samples=10, n_features=20, n_tasks=15, nnz=5)

    regressor = ReweightedMTL()
    regressor.fit(X, Y)

    start_loss = regressor.loss_history_[0]
    final_loss = regressor.loss_history_[-1]

    assert start_loss > final_loss


def test_reconstruction():
    X, Y, coef, _ = simulate_data(
        n_samples=50, n_features=250, n_tasks=25, nnz=2, corr=0, random_state=2020
    )

    alphas = np.geomspace(5e-4, 1.8e-3, num=15)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=3)

    regressor.fit(X, Y)
    coef_hat = regressor.coef_

    nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))

    assert nnz_reconstructed == 2
