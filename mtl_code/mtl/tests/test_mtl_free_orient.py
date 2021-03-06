import pytest
from itertools import product

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.utils import check_random_state

from mne.inverse_sparse.mxne_optim import mixed_norm_solver

from celer import Lasso

from mtl.simulated_data import simulate_data

from mtl.solver_free_orient import (
    MultiTaskLassoUnscaled,
    MultiTaskLassoOrientation,
)
from mtl.utils_datasets import norm_l2_inf, compute_alpha_max

####################
# Mathematical tests
####################

N_ORIENTS = [1, 3]
ACCELERATED = [False, True]
DATA_SIZE = [(10, 15), (100, 150)]
ALPHA_FRAC = [0.5, 0.1, 0.05, 0.01]


@pytest.mark.parametrize("n_samples, n_features", DATA_SIZE)
@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
@pytest.mark.parametrize("alpha_frac", ALPHA_FRAC)
def test_gap_decreasing(
    n_samples, n_features, n_orient, accelerated, alpha_frac
):
    X, Y, _, _ = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=7,
        nnz=3,
        random_state=0,
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * alpha_frac

    estimator = MultiTaskLassoUnscaled(
        alpha, n_orient, accelerated=accelerated
    )

    estimator._fit(X, Y)

    XR = X.T @ (Y - X @ estimator.coef_)
    active_set = norm(estimator.coef_, axis=1) != 0

    assert np.all(np.abs(XR) <= alpha + 1e-12), "KKT check"
    assert estimator.gap_history_[0] >= estimator.gap_history_[-1]
    assert estimator.gap_history_[-1] <= estimator.tol
    assert np.all(np.diff(estimator.gap_history_) <= 0)
    assert np.all(np.diff(estimator.primal_history_) <= 0)


@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
def test_sparsity(n_orient, accelerated):
    n_features = 60
    X, Y, _, _ = simulate_data(
        n_samples=50,
        n_features=n_features,
        n_tasks=10,
        nnz=10,
        random_state=12,
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), n_orient, copy=False)
    alpha = alpha_max * 0.1

    estimator = MultiTaskLassoUnscaled(
        alpha, n_orient, accelerated=accelerated
    )

    estimator._fit(X, Y)

    assert np.sum(norm(estimator.coef_, axis=1) == 0) > 5


####################
# Statistics tests
####################


@pytest.mark.parametrize("n_samples, n_features", DATA_SIZE)
@pytest.mark.parametrize("accelerated", ACCELERATED)
@pytest.mark.parametrize("alpha_frac", ALPHA_FRAC)
def test_single_task_fixed_orient(
    n_samples, n_features, accelerated, alpha_frac
):
    X, y, _, _ = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=1,
        nnz=3,
        random_state=340,
    )

    alpha_max = np.max(np.abs(X.T @ y))
    alpha = alpha_max * alpha_frac

    estimator1 = MultiTaskLassoUnscaled(
        alpha, 1, accelerated=accelerated, tol=1e-12
    )
    estimator2 = Lasso(alpha / len(X), fit_intercept=False, tol=1e-12)

    estimator1._fit(X, y)
    estimator2.fit(X, y)

    np.testing.assert_allclose(
        estimator1.coef_.ravel(), estimator2.coef_, rtol=1e-4
    )


@pytest.mark.parametrize("n_samples, n_features", DATA_SIZE)
@pytest.mark.parametrize("n_orient", N_ORIENTS)
@pytest.mark.parametrize("accelerated", ACCELERATED)
@pytest.mark.parametrize("alpha_frac", ALPHA_FRAC)
def test_multi_task_fixed_orient(
    n_samples, n_features, n_orient, accelerated, alpha_frac
):
    X, Y, _, _ = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=10,
        nnz=3,
        random_state=0,
    )

    alpha_max = norm_l2_inf(X.T @ Y, n_orient, copy=False)
    alpha = alpha_max * alpha_frac

    estimator = MultiTaskLassoUnscaled(
        alpha,
        n_orient,
        accelerated=accelerated,
        tol=1e-12,
        active_set_size=50,
        max_iter=10000,
    )

    estimator._fit(X, Y)

    coef, active_set, gap_history = mixed_norm_solver(
        Y, X, alpha, n_orient=n_orient, debias=False, tol=1e-12, maxit=10000
    )

    final_coef_ = np.zeros((len(active_set), 10))
    if coef is not None:
        final_coef_[active_set] = coef

    np.testing.assert_allclose(estimator.coef_, final_coef_)
