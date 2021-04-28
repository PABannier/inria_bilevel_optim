import time
from collections import defaultdict

import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state, check_X_y

from celer import MultiTaskLasso

from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max


def compute_alpha_max(X, Y):
    B = X.T @ Y
    b = norm(B, axis=1)
    return np.max(b) / X.shape[0]


def get_val(X, Y, sigma, alphas, n_iterations, random_state):
    def _init_eps_and_delta(n_samples, n_tasks):
        rng = check_random_state(random_state)
        eps = 2 * sigma / (n_samples ** 0.3)
        delta = rng.randn(n_samples, n_tasks)
        return eps, delta

    n_samples, n_tasks = Y.shape
    eps, delta = _init_eps_and_delta(n_samples, n_tasks)

    X, Y = check_X_y(X, Y, multi_output=True)
    score_grid_ = np.array([np.inf for _ in range(len(alphas))])

    coefs_grid_1, coefs_grid_2 = _fit_reweighted_with_grid(
        X, Y, n_iterations, alphas, eps, delta
    )

    for i, (coef1, coef2) in enumerate(
        zip(coefs_grid_1, coefs_grid_2)
    ):
        sure_val = _compute_sure_val(coef1, coef2, X, Y, sigma, eps, delta)
        score_grid_[i] = sure_val

    best_sure_ = np.min(score_grid_)
    best_alpha_ = alphas[np.argmin(score_grid_)]

    print(f"Best SURE: {best_sure_}")
    print(f"Best alpha: {best_alpha_}")

    return best_sure_, best_alpha_


def penalty(u):
    return 1. / (2 * np.sqrt(np.linalg.norm(u, axis=1)) + np.finfo(float).eps)


def _reweight_op(regressor, X, Y, w):
    X_w = X / w[np.newaxis, :]
    regressor.fit(X_w, Y)

    coef = (regressor.coef_ / w).T
    w = penalty(coef)

    return coef, w


def _compute_sure_val(coef1, coef2, X, Y, sigma, eps, delta):

    n_samples, n_features = X.shape
    n_samples, n_tasks = Y.shape

    # Note: Celer returns the transpose of the coefficient
    # matrix
    if coef1.shape[0] != X.shape[1]:
        coef1 = coef1.T
        coef2 = coef2.T

    # Compute the dof
    dof = ((X @ (coef2 - coef1)) * delta).sum() / eps
    # compute the SURE
    sure = norm(Y - X @ coef1) ** 2
    sure -= n_samples * n_tasks * sigma ** 2
    sure += 2 * dof * sigma ** 2

    return sure


def _fit_reweighted_with_grid(X, Y, n_iterations, alphas, eps, delta):
    _, n_features = X.shape
    _, n_tasks = Y.shape
    n_alphas = len(alphas)

    coef1_0 = np.empty((n_alphas, n_features, n_tasks))
    coef2_0 = np.empty((n_alphas, n_features, n_tasks))

    assert np.all(np.diff(alphas) < 0)

    Y_eps = Y + eps * delta

    # Warm start first iteration
    regressor1 = MultiTaskLasso(np.nan, fit_intercept=False, warm_start=True)
    regressor2 = MultiTaskLasso(np.nan, fit_intercept=False, warm_start=True)

    # Copy grid of first iteration (leverages convexity)
    for j, alpha in enumerate(alphas):
        regressor1.alpha = alpha
        regressor2.alpha = alpha
        coef1_0[j] = regressor1.fit(X, Y).coef_.T
        coef2_0[j] = regressor2.fit(X, Y_eps).coef_.T

    regressor1.warm_start = False
    regressor2.warm_start = False

    coefs_1_ = coef1_0.copy()
    coefs_2_ = coef2_0.copy()

    for j, alpha in enumerate(alphas):
        regressor1.alpha = alpha
        regressor2.alpha = alpha

        w1 = penalty(coef1_0[j])
        w2 = penalty(coef2_0[j])

        for _ in range(n_iterations - 1):
            mask1 = (w1 != 1. / np.finfo(float).eps)
            mask2 = (w2 != 1. / np.finfo(float).eps)
            coefs_1_[j][~mask1] = 0.
            coefs_2_[j][~mask2] = 0.

            if mask1.sum():
                coefs_1_[j][mask1], w1[mask1] = _reweight_op(regressor1, X[:, mask1], Y, w1[mask1])
            if mask2.sum():
                coefs_2_[j][mask2], w2[mask2] = _reweight_op(regressor2, X[:, mask2], Y_eps, w2[mask2])

    return coefs_1_, coefs_2_


if __name__ == "__main__":
    random_state = 42

    N_SAMPLES = 102
    N_FEATURES = 100
    N_TASKS = 100
    NNZ = 4

    # N_SAMPLES = 10
    # N_FEATURES = 20
    # N_TASKS = 2
    # NNZ = 2

    n_alphas = 100
    n_iterations = 5

    X, Y, _, sigma = simulate_data(
        N_SAMPLES, N_FEATURES, N_TASKS, NNZ, random_state=random_state
    )

    max_alpha = compute_alpha_max(X, Y)
    print("alpha max", max_alpha)
    alphas = np.geomspace(max_alpha, max_alpha / 30, n_alphas)

    start_time = time.time()
    best_sure_, best_alpha_ = \
        get_val(X, Y, sigma, alphas, n_iterations=n_iterations, random_state=random_state)
    print("Duration (with warm start):", time.time() - start_time)

    criterion = SURE(ReweightedMultiTaskLasso, sigma, random_state=random_state)
    start_time = time.time()
    best_sure, best_alpha = np.inf, None
    for alpha in alphas:
        sure_val = criterion.get_val(
            X, Y, alpha, warm_start=False, verbose=False,
            n_iterations=n_iterations
        )
        if sure_val < best_sure:
            best_sure = sure_val
            best_alpha = alpha
    print(f"Best SURE: {best_sure}")
    print(f"Best alpha: {best_alpha}")
    print("Duration (without warm start):", time.time() - start_time)
