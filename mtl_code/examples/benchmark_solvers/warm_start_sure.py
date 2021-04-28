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


def get_val(X, Y, sigma, alpha_grid, random_state):
    def _init_eps_and_delta(n_samples, n_tasks):
        rng = check_random_state(random_state)
        eps = 2 * sigma / (n_samples ** 0.3)
        delta = rng.randn(n_samples, n_tasks)
        return eps, delta

    n_samples, n_tasks = Y.shape
    eps, delta = _init_eps_and_delta(n_samples, n_tasks)

    X, Y = check_X_y(X, Y, multi_output=True)
    score_grid_ = np.array([np.inf for _ in range(len(alpha_grid))])

    coefs_grid_1, coefs_grid_2 = _fit_reweighted_with_grid(
        X, Y, 5, alpha_grid, eps, delta
    )

    for i, (coef1, coef2) in enumerate(
        zip(coefs_grid_1.values(), coefs_grid_2.values())
    ):
        sure_val = _compute_sure_val(coef1, coef2, X, Y, sigma, eps, delta)
        score_grid_[i] = sure_val

    best_sure_ = np.min(score_grid_)
    best_alpha_ = alpha_grid[np.argmin(score_grid_)]

    print(f"Best SURE: {best_sure_}")
    print(f"Best alpha: {best_alpha_}")

    return best_sure_, best_alpha_


def _reweight_op(regressor, X, Y, w):
    penalty = lambda u: 1 / (
        2 * np.sqrt(np.linalg.norm(u, axis=1)) + np.finfo(float).eps
    )
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
    dof = ((X @ coef2 - X @ coef1) * delta).sum() / eps
    # compute the SURE
    sure = norm(Y - X @ coef1) ** 2
    sure -= n_samples * n_tasks * sigma ** 2
    sure += 2 * dof * sigma ** 2

    return sure


def _fit_reweighted_with_grid(X, Y, n_iterations, alpha_grid, eps, delta):
    n_samples, n_features = X.shape
    n_samples, n_tasks = Y.shape

    coefs_1_ = dict()
    coefs_2_ = dict()

    weights_1_ = defaultdict(lambda: np.ones(n_features))
    weights_2_ = defaultdict(lambda: np.ones(n_features))

    for _ in range(n_iterations):
        regressor = MultiTaskLasso(0.1, fit_intercept=False, warm_start=True)
        for j, alpha in enumerate(alpha_grid):
            regressor.alpha = alpha

            coef1, w1 = _reweight_op(regressor, X, Y, weights_1_[j])
            coef2, w2 = _reweight_op(
                regressor, X, Y + eps * delta, weights_2_[j]
            )

            coefs_1_[j], weights_1_[j] = coef1, w1
            coefs_2_[j], weights_2_[j] = coef2, w2

    return coefs_1_, coefs_2_


if __name__ == "__main__":
    random_state = 42

    N_SAMPLES = 102
    N_FEATURES = 1000
    N_TASKS = 100
    NNZ = 4

    X, Y, _, sigma = simulate_data(N_SAMPLES, N_FEATURES, N_TASKS, NNZ)

    max_alpha = compute_alpha_max(X, Y)
    print("alpha max", max_alpha)
    alphas = np.geomspace(max_alpha, max_alpha / 30, 100)

    print("\n")

    start_time = time.time()
    get_val(X, Y, sigma, alphas, random_state)
    print("Duration (with warm start):", time.time() - start_time)

    criterion = SURE(ReweightedMultiTaskLasso, sigma)
    start_time = time.time()
    best_sure, best_alpha = np.inf, None
    for alpha in alphas:
        sure_val = criterion.get_val(
            X, Y, alpha, warm_start=False, verbose=False
        )
        if sure_val < best_sure:
            best_sure = sure_val
            best_alpha = alpha
    print("Duration (without warm start):", time.time() - start_time)
