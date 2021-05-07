import pytest

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max

####################
# Mathematical tests
####################


corr_coeffs = [0, 0.3, 0.5, 0.7, 0.9]


@pytest.mark.parametrize("corr", corr_coeffs)
def test_training_loss_decrease(corr):
    X, Y, W, _ = simulate_data(
        n_samples=10, n_features=20, n_tasks=15, nnz=5, corr=corr
    )

    regressor = ReweightedMultiTaskLasso()
    regressor.fit(X, Y)

    start_loss = regressor.loss_history_[0]
    final_loss = regressor.loss_history_[-1]

    assert start_loss > final_loss


@pytest.mark.parametrize("corr", corr_coeffs)
def test_decreasing_loss_every_step(corr):
    X, Y, W, _ = simulate_data(
        n_samples=10,
        n_features=20,
        n_tasks=15,
        nnz=5,
        corr=corr,
        random_state=2020,
    )

    regressor = ReweightedMultiTaskLasso()
    regressor.fit(X, Y)

    diffs = np.diff(regressor.loss_history_)

    print(diffs)

    assert np.all(diffs <= 5e-5)


@pytest.mark.parametrize("corr", corr_coeffs)
def test_decreasing_loss_every_step_free_orientation(corr):
    X, Y, W, _ = simulate_data(
        n_samples=10,
        n_features=21,
        n_tasks=15,
        nnz=5,
        corr=corr,
        random_state=42,
    )

    regressor = ReweightedMultiTaskLasso(n_orient=3)
    regressor.fit(X, Y)

    diffs = np.diff(regressor.loss_history_)

    print(diffs)

    assert np.all(diffs <= 5e-5)


####################
# Statistics tests
####################


def test_reconstruction():
    X, Y, coef, _ = simulate_data(
        n_samples=5,
        n_features=10,
        n_tasks=8,
        nnz=2,
        random_state=2020,
    )

    alpha_max = compute_alpha_max(X, Y)
    lb = alpha_max * 0.1
    hb = alpha_max * 0.5

    alphas = np.geomspace(lb, hb, num=15)
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=5)

    regressor.fit(X, Y)
    coef_hat = regressor.coef_

    nnz_reconstructed = np.count_nonzero(np.count_nonzero(coef_hat, axis=1))

    assert np.abs(nnz_reconstructed - 3) <= 1
