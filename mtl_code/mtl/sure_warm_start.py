from collections import defaultdict
from tqdm import tqdm
import time

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state, check_X_y

from celer import MultiTaskLasso


class SUREForReweightedMultiTaskLasso:
    def __init__(
        self,
        sigma,
        alpha_grid,
        n_iterations=5,
        penalty=None,
        random_state=None,
    ):
        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)

        self.eps = None
        self.delta = None

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1.0 / (
                2 * np.sqrt(np.linalg.norm(u, axis=1)) + np.finfo(float).eps
            )

    def get_val(self, X, Y):
        n_samples, n_tasks = Y.shape

        if self.eps is None or self.delta is None:
            self._init_eps_and_delta(n_samples, n_tasks)

        X, Y = check_X_y(X, Y, multi_output=True)
        score_grid_ = np.array([np.inf for _ in range(self.n_alphas)])

        coefs_grid_1, coefs_grid_2 = self._fit_reweighted_with_grid(X, Y)

        for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
            sure_val = self._compute_sure_val(coef1, coef2, X, Y)
            score_grid_[i] = sure_val

        best_sure_ = np.min(score_grid_)
        best_alpha_ = self.alpha_grid[np.argmin(score_grid_)]

        return best_sure_, best_alpha_

    def _reweight_op(self, regressor, X, Y, w):
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, Y)

        coef = (regressor.coef_ / w).T
        w = self.penalty(coef)

        return coef, w

    def _compute_sure_val(self, coef1, coef2, X, Y):

        n_samples, n_features = X.shape
        n_samples, n_tasks = Y.shape

        # Note: Celer returns the transpose of the coefficient
        # matrix
        if coef1.shape[0] != X.shape[1]:
            coef1 = coef1.T
            coef2 = coef2.T

        # Compute the dof
        dof = ((X @ (coef2 - coef1)) * self.delta).sum() / self.eps
        # compute the SURE
        sure = norm(Y - X @ coef1) ** 2
        sure -= n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure

    def _fit_reweighted_with_grid(self, X, Y):
        _, n_features = X.shape
        _, n_tasks = Y.shape
        n_alphas = len(self.alpha_grid)

        coef1_0 = np.empty((n_alphas, n_features, n_tasks))
        coef2_0 = np.empty((n_alphas, n_features, n_tasks))

        Y_eps = Y + self.eps * self.delta

        # Warm start first iteration
        regressor1 = MultiTaskLasso(
            np.nan, fit_intercept=False, warm_start=True
        )
        regressor2 = MultiTaskLasso(
            np.nan, fit_intercept=False, warm_start=True
        )

        # Copy grid of first iteration (leverages convexity)
        print("First iteration")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor1.alpha = alpha
            regressor2.alpha = alpha
            coef1_0[j] = regressor1.fit(X, Y).coef_.T
            coef2_0[j] = regressor2.fit(X, Y_eps).coef_.T

        regressor1.warm_start = False
        regressor2.warm_start = False

        coefs_1_ = coef1_0.copy()
        coefs_2_ = coef2_0.copy()

        print("Next iterations...")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor1.alpha = alpha
            regressor2.alpha = alpha

            w1 = self.penalty(coef1_0[j])
            w2 = self.penalty(coef2_0[j])

            for _ in range(self.n_iterations - 1):
                mask1 = w1 != 1.0 / np.finfo(float).eps
                mask2 = w2 != 1.0 / np.finfo(float).eps
                coefs_1_[j][~mask1] = 0.0
                coefs_2_[j][~mask2] = 0.0

                if mask1.sum():
                    coefs_1_[j][mask1], w1[mask1] = self._reweight_op(
                        regressor1, X[:, mask1], Y, w1[mask1]
                    )
                if mask2.sum():
                    coefs_2_[j][mask2], w2[mask2] = self._reweight_op(
                        regressor2, X[:, mask2], Y_eps, w2[mask2]
                    )

        return coefs_1_, coefs_2_

    def _init_eps_and_delta(self, n_samples, n_tasks):
        rng = check_random_state(self.random_state)
        self.eps = 2 * self.sigma / (n_samples ** 0.3)
        self.delta = rng.randn(n_samples, n_tasks)
