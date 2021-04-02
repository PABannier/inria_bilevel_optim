import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state

from mtl.mtl import ReweightedMultiTaskLasso


class SURE:
    """Stein Unbiased Risk Estimator (SURE) implementation
    for Multi-Task LASSO problems.

    Implements the finite-difference Monte-Carlo approximation
    of the SURE for Multi-Task LASSO.

    Parameters
    ----------
    sigma: float
        Noise level.

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

    def __init__(self, sigma, random_state=None):
        self.sigma = sigma
        self.rng = check_random_state(random_state)

        self.eps = None
        self.delta = None

    def get_val(self, X, Y, alpha, n_iterations=10):
        """Performs the double forward step used in finite differences
        and evaluates an Monte-Carlo finite-difference approximation of
        the SURE.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y: np.ndarray of shape (n_samples, n_tasks)
            Target matrix.

        alpha: float
            Regularizing constant.

        n_iterations: int, default=10
            Number of reweighting steps.

        Returns
        -------
        val: float
            Monte-Carlo Finite Difference SURE approximation.
        """
        n_samples, n_tasks = Y.shape

        if self.delta is None or self.eps is None:
            self.init_eps_and_delta(n_samples, n_tasks)

        # fit 2 models in Y and Y + epsilon * delta
        model = ReweightedMultiTaskLasso(alpha, n_iterations, verbose=False)
        model.fit(X, Y)
        coef1 = model.coef_
        model.fit(X, Y + self.eps * self.delta)
        coef2 = model.coef_

        # compute the dof
        dof = ((X @ coef2 - X @ coef1) * self.delta).sum() / self.eps
        # compute the SURE
        sure = norm(Y - X @ coef1) ** 2
        sure -= n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure

    def init_eps_and_delta(self, n_features, n_tasks):
        """Implements a heuristic found by [1] to correctly
        set epsilon, and initializes delta with an isotropic
        Gaussian distribution.

        Parameters
        ----------
        n_features: int
            Number of features in the design matrix.
        """
        self.eps = 2 * self.sigma / (n_features ** 0.3)
        self.delta = self.rng.randn(n_features, n_tasks)
