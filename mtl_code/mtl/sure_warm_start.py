from collections import defaultdict
from itertools import product
from tqdm import tqdm
import time

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state, check_X_y

from celer import MultiTaskLasso

from mtl.solver_free_orient import MultiTaskLassoOrientation


class SUREForReweightedMultiTaskLasso:
    def __init__(
        self,
        sigma,
        alpha_grid,
        n_iterations=5,
        penalty=None,
        n_orient=1,
        random_state=None,
    ):
        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_iterations = n_iterations
        self.n_orient = n_orient
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)

        self.sure_path_ = np.empty(self.n_alphas)
        self.dof_history_ = np.empty(self.n_alphas)
        self.data_fitting_history_ = np.empty(self.n_alphas)

        self.eps = None
        self.delta = None

        if self.n_orient <= 0:
            raise ValueError("Number of orientations can't be negative.")

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = self._penalty

    def get_val(self, X, Y):
        n_samples, n_tasks = Y.shape

        if self.eps is None or self.delta is None:
            self._init_eps_and_delta(n_samples, n_tasks)

        X, Y = check_X_y(X, Y, multi_output=True)

        coefs_grid_1, coefs_grid_2 = self._fit_reweighted_with_grid(X, Y)

        for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
            sure_val, dof_term, data_fitting_term = self._compute_sure_val(
                coef1, coef2, X, Y
            )
            self.sure_path_[i] = sure_val
            self.dof_history_[i] = dof_term
            self.data_fitting_history_[i] = data_fitting_term

        best_sure_ = np.min(self.sure_path_)
        best_alpha_ = self.alpha_grid[np.argmin(self.sure_path_)]

        return best_sure_, best_alpha_

    def _reweight_op(self, regressor, X, Y, w):
        X_w = X / np.repeat(w[np.newaxis, :], self.n_orient)
        regressor.fit(X_w, Y)

        if self.n_orient == 1:
            coef = (regressor.coef_ / w).T
        else:
            coef = (
                regressor.coef_.T / np.repeat(w[np.newaxis, :], self.n_orient)
            ).T

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
        df_term = norm(Y - X @ coef1) ** 2
        sure = df_term - n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure, dof, df_term

    def _fit_reweighted_with_grid(self, X, Y):
        _, n_features = X.shape
        _, n_tasks = Y.shape
        n_alphas = len(self.alpha_grid)

        coef1_0 = np.empty((n_alphas, n_features, n_tasks))
        coef2_0 = np.empty((n_alphas, n_features, n_tasks))

        Y_eps = Y + self.eps * self.delta

        # Warm start first iteration
        if self.n_orient == 1:
            regressor1 = MultiTaskLasso(
                np.nan, fit_intercept=False, warm_start=True
            )
            regressor2 = MultiTaskLasso(
                np.nan, fit_intercept=False, warm_start=True
            )
        else:
            regressor1 = MultiTaskLassoOrientation(
                np.nan, warm_start=True, n_orient=self.n_orient
            )
            regressor2 = MultiTaskLassoOrientation(
                np.nan, warm_start=True, n_orient=self.n_orient
            )

        # Copy grid of first iteration (leverages convexity)
        print("First iteration")
        for j, alpha in tqdm(enumerate(self.alpha_grid), total=self.n_alphas):
            regressor1.alpha = alpha
            regressor2.alpha = alpha

            if self.n_orient == 1:
                coef1_0[j] = regressor1.fit(X, Y).coef_.T
                coef2_0[j] = regressor2.fit(X, Y_eps).coef_.T
            else:
                coef1_0[j] = regressor1.fit(X, Y).coef_
                coef2_0[j] = regressor2.fit(X, Y_eps).coef_

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

                mask1_full = np.repeat(mask1, self.n_orient)
                mask2_full = np.repeat(mask2, self.n_orient)

                coefs_1_[j][~mask1_full] = 0.0
                coefs_2_[j][~mask2_full] = 0.0

                if mask1.sum():
                    coefs_1_[j][mask1_full], w1[mask1] = self._reweight_op(
                        regressor1, X[:, mask1_full], Y, w1[mask1]
                    )
                if mask2.sum():
                    coefs_2_[j][mask2_full], w2[mask2] = self._reweight_op(
                        regressor2, X[:, mask2_full], Y_eps, w2[mask2]
                    )

        return coefs_1_, coefs_2_

    def _init_eps_and_delta(self, n_samples, n_tasks):
        rng = check_random_state(self.random_state)
        self.eps = 2 * self.sigma / (n_samples ** 0.3)
        self.delta = rng.randn(n_samples, n_tasks)

    def _penalty(self, coef):
        """Defines a non-convex penalty for reweighting
        the design matrix from the regression coefficients.

        Takes into account the number of orientations
        of the problem.

        Parameters
        ----------
        coef : array of shape (n_features, n_times)
            Coefficient matrix.

        Returns
        -------
        penalty : array of shape (n_positions,)
            Penalty vector.
        """
        n_positions = coef.shape[0] // self.n_orient
        coef = coef.reshape(n_positions, -1)
        m_norm = np.sqrt(norm(coef, axis=1))
        return 1 / (2 * m_norm + np.finfo(float).eps)
