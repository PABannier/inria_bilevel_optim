import pytest

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

from mne.inverse_sparse.mxne_optim import mixed_norm_solver

from celer import Lasso

from mtl.simulated_data import simulate_data

from solver_lasso.solver_free_orient import MultiTaskLassoOrientation
from solver_lasso.utils import norm_l2_inf

####################
# Mathematical tests
####################

if __name__ == "__main__":
    X, Y, _, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=7, nnz=3, random_state=0
    )

    n_orient = 3
    accelerated = False

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoOrientation(
        alpha, n_orient, accelerated=accelerated
    )

    estimator.fit(X, Y)

    plt.plot(np.log10(estimator.gap_history_))
    plt.show()


"""

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

    assert np.sum(norm(estimator.coef_, axis=1) == 0) > 10


####################
# Statistics tests
####################


@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_single_task_fixed_orient(accelerated):
    X, y, _, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=1, nnz=3, random_state=0
    )

    alpha_max = np.max(np.abs(X.T @ y))
    alpha = alpha_max * 0.1

    estimator1 = MultiTaskLassoOrientation(alpha, 1, accelerated=accelerated)
    estimator2 = Lasso(alpha / len(X), fit_intercept=False)

    estimator1.fit(X, y)
    estimator2.fit(X, y)

    np.testing.assert_allclose(
        estimator1.coef_.ravel(), estimator2.coef_, rtol=1e-4
    )


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_multi_task_fixed_orient(n_orient, accelerated):
    X, Y, _, _ = simulate_data(
        n_samples=50, n_features=60, n_tasks=10, nnz=3, random_state=0
    )

    alpha_max = norm_l2_inf(X.T @ Y, n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoOrientation(
        alpha, n_orient, accelerated=accelerated, tol=1e-8, active_set_size=50
    )

    estimator.fit(X, Y)

    coef, active_set, gap_history = mixed_norm_solver(
        Y,
        X,
        alpha,
        n_orient=n_orient,
        debias=False,
    )

    final_coef_ = np.zeros((len(active_set), 10))
    if coef is not None:
        final_coef_[active_set] = coef

    np.testing.assert_allclose(estimator.coef_, final_coef_)
"""
