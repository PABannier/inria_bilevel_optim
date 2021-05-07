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


N_ORIENTS = [1, 3]
DATA_SIZE = [(10, 15), (50, 75)]
ALPHA_FRAC = [0.5, 0.1, 0.05, 0.01]


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("n_samples, n_features", DATA_SIZE)
@pytest.mark.parametrize("alpha_frac", ALPHA_FRAC)
def test_training_loss_decrease(n_orient, n_samples, n_features, alpha_frac):
    X, Y, _, _ = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=15,
        nnz=5,
        random_state=0,
    )

    alpha_max = compute_alpha_max(X, Y)
    alpha = alpha_max * alpha_frac

    regressor = ReweightedMultiTaskLasso(
        alpha, n_orient=n_orient, tol=1e-6, n_iterations=5
    )
    regressor.fit(X, Y)

    start_loss = regressor.loss_history_[0]
    final_loss = regressor.loss_history_[-1]

    assert start_loss > final_loss


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("n_samples, n_features", DATA_SIZE)
@pytest.mark.parametrize("alpha_frac", ALPHA_FRAC)
def test_decreasing_loss_every_step(
    n_orient, n_samples, n_features, alpha_frac
):
    X, Y, _, _ = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=15,
        nnz=5,
        random_state=2020,
    )

    alpha_max = compute_alpha_max(X, Y)
    alpha = alpha_max * alpha_frac

    regressor = ReweightedMultiTaskLasso(
        alpha, n_orient=n_orient, tol=1e-6, n_iterations=5
    )
    regressor.fit(X, Y)

    diffs = np.diff(regressor.loss_history_)
    print(diffs[diffs > 1e-8])
    np.testing.assert_array_less(diffs, 1e-5)
