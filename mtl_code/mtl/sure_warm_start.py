from collections import defaultdict

import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state, check_X_y

from celer import MultiTaskLasso
from mtl.mtl import ReweightedMultiTaskLasso


class SUREForReweightedMultiTaskLasso:
    """Stein Unbiased Risk Estimator (SURE) implementation
    for Multi-Task LASSO problems with warm start.

    Implements the finite-difference Monte-Carlo approximation
    of the SURE for Multi-Task LASSO.

    This class is specifically designed for Reweighted Multi-Task
    Lasso to include warm start. The implementation required for
    warm starting reweighted LASSO models is different than non-
    reweighted models. Indeed, we can only warm start at the first
    iteration since the weights change between alphas and can't be
    reused.

    Parameters
    ----------
    sigma: float
        Noise level.

    alpha_grid: array
        The alpha grid to test.

    penalty: callable
        Penalty to apply for reweighted the weights in Reweighted
        Multi-Task LASSO.

    n_iterations: int
        Number of reweighting iterations for Reweighted Multi-Task
        LASSO.

    random_state: int, RandomState instance, default=None
        The seed of the pseudo-random number generator.

    Attributes
    ----------
    rng: RandomState
        Random number generator.

    eps: float
        Epsilon value used for finite difference.

    delta: np.ndarray of shape (n_features, n_features)
        Random matrix whose columns are used as directions
        to compute directional derivatives for Monte-Carlo SURE.

    References
    ----------
    .. [1] C.-A. Deledalle, Stein Unbiased GrAdient estimator of the Risk
    (SUGAR) for multiple parameter selection.
    SIAM J. Imaging Sci., 7(4), 2448-2487.

    """

    def __init__(
        self,
        sigma,
        alpha_grid,
        n_iterations=5,
        penalty=None,
        random_state=None,
    ):
        self.estimator = MultiTaskLasso()
        self.sigma = sigma
        self.rng = check_random_state(random_state)

        self.eps = None
        self.delta = None

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1 / (
                2 * np.sqrt(norm(u, axis=1)) + np.finfo(float).eps
            )

    def get_val(self, X, Y):
        """Performs the double forward step used in finite differences
        and evaluates an Monte-Carlo finite-difference approximation of
        the SURE.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y: np.ndarray of shape (n_samples, n_tasks)
            Target matrix.

        Returns
        -------
        val: float
            Monte-Carlo Finite Difference SURE approximation.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        score_grid_ = np.array([np.inf for _ in range(len(self.alpha_grid))])

        coefs_grid_1, coefs_grid_2 = self._fit_reweighted_with_grid(X, Y)

        for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
            sure_val = compute_sure_val(coef1, coef2, X, Y)
            score_grid_[i] = sure_val

        self.best_sure_ = np.min(score_grid_)
        self.best_alpha = self.alpha_grid[np.argmin(score_grid_)]

        print(f"Best SURE: {self.best_sure_}")
        print(f"Best alpha: {self.best_alpha_}")

    def _fit_reweighted_with_grid(self, X, Y):
        n_samples, n_features = X.shape
        n_samples, n_tasks = Y.shape

        if self.delta is None or self.eps is None:
            self._init_eps_and_delta(n_samples, n_tasks)

        coefs_1_ = dict()
        coefs_2_ = dict()

        weights_1_ = defaultdict(lambda: np.ones(n_features))
        weights_2_ = defaultdict(lambda: np.ones(n_features))

        for _ in range(self.n_iterations):
            regressor = MultiTaskLasso(
                0.1, fit_intercept=False, warm_start=True
            )
            for j, alpha in enumerate(self.alpha_grid):
                regressor.alpha = alpha

                coef1, w1 = self._reweight_op(regressor, X, Y, weights_1_[j])
                coef2, w2 = self._reweight_op(
                    regressor, X, Y + self.eps * self.delta, weights_2_[j]
                )

                coefs_1_[j], weights_1_[j] = coef1, w1
                coefs_2_[j], weights_2_[j] = coef2, w2

        return coefs_1_, coefs_2_

    def compute_sure_val(self, coef1, coef2, X, Y):
        n_samples, n_features = X.shape
        n_samples, n_tasks = Y.shape

        # Note: Celer returns the transpose of the coefficient
        # matrix
        if coef1.shape[0] != X.shape[1]:
            coef1 = coef1.T
            coef2 = coef2.T

        # compute the dof
        dof = ((X @ coef2 - X @ coef1) * self.delta).sum() / self.eps
        # compute the SURE
        sure = norm(Y - X @ coef1) ** 2
        sure -= n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure

    def _reweight_op(self, regressor, X, Y, w):
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, Y)

        coef = (regressor.coef_ / w).T
        w = self.penalty(coef)

        return coef, w

    def _init_eps_and_delta(self, n_samples, n_tasks):
        """Implements a heuristic found by [1] to correctly
        set epsilon, and initializes delta with an isotropic
        Gaussian distribution.

        Parameters
        ----------
        n_samples: int
            Number of samples in the design matrix.

        n_tasks: int
            Number of tasks in the problem.
        """
        self.eps = 2 * self.sigma / (n_samples ** 0.3)
        self.delta = self.rng.randn(n_samples, n_tasks)
