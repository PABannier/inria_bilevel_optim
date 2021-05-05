import pytest

import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state

from mtl.simulated_data import simulate_data

from solver_lasso.solver_free_orient import MultiTaskLassoOrientation
from solver_lasso.utils import norm_l2_inf

N_ORIENTS = [1, 3]
ACCELERATED = [False, True]


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_loss_decreasing(n_orient, accelerated):
    X, Y, _, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=7, nnz=3, random_state=0
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoOrientation(
        alpha, n_orient, accelerated=accelerated
    )

    estimator.fit(X, Y)

    assert estimator.gap_history_[0] > estimator.gap_history_[-1]


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_loss_decreasing_every_iteration(n_orient, accelerated):
    X, Y, _, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=7, nnz=3, random_state=0
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoOrientation(
        alpha, n_orient, accelerated=accelerated
    )

    estimator.fit(X, Y)

    diffs = np.diff(estimator.gap_history_)

    assert np.all(diffs < 1e-3)


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_sparsity(n_orient, accelerated):
    n_features = 60
    X, Y, _, _ = simulate_data(
        n_samples=50, n_features=n_features, n_tasks=10, nnz=10, random_state=0
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoOrientation(
        alpha, n_orient, accelerated=accelerated
    )

    estimator.fit(X, Y)

    assert n_features - estimator.coef_.shape[0] > 10
