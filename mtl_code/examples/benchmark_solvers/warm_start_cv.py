from collections import defaultdict
import time
import ipdb

import numpy as np
from numpy.linalg import norm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from celer import MultiTaskLasso

from mtl.cross_validation import ReweightedMultiTaskLassoCV

N_SAMPLES = 20
N_FEATURES = 40
N_TASKS = 10


def compute_alpha_max(X, Y):
    B = X.T @ Y
    b = norm(B, axis=1)
    return np.max(b) / X.shape[0]


def reweight_op(regressor, X, Y, w):
    penalty = lambda u: 1 / (
        2 * np.sqrt(np.linalg.norm(u, axis=1)) + np.finfo(float).eps
    )
    X_w = X / w[np.newaxis, :]
    regressor.fit(X_w, Y)

    coef = (regressor.coef_ / w).T
    w = penalty(coef)

    return coef, w


def fit_reweighted_with_grid(X, Y, alpha_grid, n_iter=5, warm_start=True):
    coefs = dict()
    weights = defaultdict(lambda: np.ones(N_FEATURES))
    losses = np.empty((n_iter, len(alpha_grid)))

    for i in range(n_iter):
        regressor = MultiTaskLasso(
            0.1, fit_intercept=False, warm_start=warm_start
        )
        for j, alpha in enumerate(alpha_grid):
            regressor.alpha = alpha

            objective = lambda W: np.sum((Y - X @ W) ** 2) / (
                2 * N_SAMPLES
            ) + alpha * np.sum(np.sqrt(norm(W, axis=1)))

            coef, w = reweight_op(regressor, X, Y, weights[j])

            coefs[j] = coef
            weights[j] = w
            losses[i][j] = objective(coef)

    return coefs, losses


def cross_val(X, Y, alpha_grid, random_state, warm_start=True):
    kf = KFold(5, random_state=random_state, shuffle=True)

    scores = [np.inf for _ in range(len(alpha_grid))]
    Y_oofs = [np.zeros((N_SAMPLES, N_TASKS)) for _ in range(len(alpha_grid))]

    for trn_idx, val_idx in kf.split(X, Y):
        X_train, Y_train = X[trn_idx, :], Y[trn_idx, :]
        X_valid, Y_valid = X[val_idx, :], Y[val_idx, :]

        coefs, losses = fit_reweighted_with_grid(
            X_train, Y_train, alpha_grid, warm_start=warm_start
        )

        # diffs = np.diff(losses, axis=0)
        # assert np.all(diffs <= 1e-5), "Objective did not decrease"

        predictions = [X_valid @ coef for coef in coefs.values()]

        for i in range(len(Y_oofs)):
            Y_oofs[i][val_idx, :] = predictions[i]

    for i in range(len(Y_oofs)):
        scores[i] = mean_squared_error(Y, Y_oofs[i])

    print("best score:", np.min(scores))
    print("best alpha:", alpha_grid[np.argmin(scores)])


if __name__ == "__main__":
    random_state = 42
    rng = np.random.default_rng(random_state)
    X = rng.random((N_SAMPLES, N_FEATURES))
    Y = rng.random((N_SAMPLES, N_TASKS))

    max_alpha = compute_alpha_max(X, Y)
    print("alpha max", max_alpha)
    alphas = np.geomspace(max_alpha, max_alpha / 30, 50)

    print("\n")

    cv_regressor = ReweightedMultiTaskLassoCV(alphas)
    start = time.time()
    cv_regressor.fit(X, Y)
    print(f"Warm start=True, {time.time() - start:.2f}s")

    print("\n")

    cv_regressor = ReweightedMultiTaskLassoCV(alphas, warm_start=False)
    start = time.time()
    cv_regressor.fit(X, Y)
    print(f"Warm start=False, {time.time() - start:.2f}s")

    print("\n")
    print("=" * 30)
    print("\n")

    start = time.time()
    cross_val(X, Y, alphas, random_state, warm_start=True)
    print(f"Warm start=True, {time.time() - start:.2f}s")

    print("\n")

    start = time.time()
    cross_val(X, Y, alphas, random_state, warm_start=False)
    print(f"Warm start=False, {time.time() - start:.2f}s")
