import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state

from celer import MultiTaskLasso
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

    def __init__(self, estimator_factory, sigma, random_state=None):
        self.estimator_factory = estimator_factory
        self.sigma = sigma
        self.rng = check_random_state(random_state)

        self.eps = None
        self.delta = None

    def get_val(self, X, Y, alpha, n_iterations=5, **estimator_kwargs):
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
        model1 = self.estimator_factory(alpha, n_iterations, **estimator_kwargs)
        model2 = self.estimator_factory(alpha, n_iterations, **estimator_kwargs)
        model1.fit(X, Y)
        coef1 = model1.coef_
        Y_eps = Y + self.eps * self.delta
        model2.fit(X, Y_eps)
        coef2 = model2.coef_

        # Note: Celer returns the transpose of the coefficient
        # matrix
        if coef1.shape[0] != X.shape[1]:
            coef1 = coef1.T
            coef2 = coef2.T

        # compute the dof
        dof = (X @ (coef2 - coef1) * self.delta).sum() / self.eps
        # compute the SURE
        sure = norm(Y - X @ coef1) ** 2
        sure -= n_samples * n_tasks * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure

    def init_eps_and_delta(self, n_samples, n_tasks):
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
