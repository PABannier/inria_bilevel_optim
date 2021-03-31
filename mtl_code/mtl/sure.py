import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state

from mtl_code.mtl import ReweightedMTL


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
        n_features, n_tasks = Y.shape

        if self.delta is None or self.eps is None:
            self.init_eps_and_delta(n_features)

        mask, dense = self.forward_lasso(X, Y, alpha, n_iterations)
        mask2, dense2 = self.forward_lasso(
            X, Y + self.eps * self.delta, alpha, n_iterations
        )

        val = self.get_val_outer(X, Y, mask, dense, mask2, weight2)
        return val

    def get_val_outer(self, X, Y, mask, weight, mask2, weight2):
        """Computes the degrees of freedom and the SURE.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y: np.ndarray of shape (n_samples, n_tasks)
            Target matrix.

        mask: np.ndarray of shape (n_features, n_tasks)
            Boolean array corresponding to the non-zero
            coefficients of the solution of the first
            optimization problem.

        dense: np.ndarray
            Values of the non-zero coefficients of the
            solution of the first optimization problem.

        mask2: np.ndarray of shape (n_features, n_tasks)
            Boolean array corresponding to the non-zero
            coefficients of the solution of the second
            optimization problem.

        dense2: np.ndarray
            Values of the non-zero coefficients of the
            solution of the second optimization problem.

        Returns
        -------
        val: float
            SURE approximation using Monte-Carlo finite differences.
        """
        X_m = X[:, mask]
        dof = (X[:, mask2] @ dense2 - X_m @ dense) @ self.delta  # Where is + epsilon?
        dof /= self.epsilon

        val = norm(Y - X_m @ dense) ** 2
        val -= n_features * n_tasks * self.sigma ** 2
        val += 2 * n_tasks * dof * self.sigma ** 2
        return val

    def forward(self, X, Y, alpha, n_iterations):
        """Solves the Reweighted Multi-Task LASSO problem
        for X, Y and parameters alpha and n_iterations.

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
        mask: np.ndarray
            Boolean array corresponding to the support.

        dense: np.ndarray
            Values of the non-zero coefficients of the
            solution of the optimization problem.
        """
        model = ReweightedMTL(alpha, n_iterations, verbose=False)
        model.fit(X, Y)

        mask = model.weights[:, 0] != 0
        dense = model.weights[mask, :]
        return mask, dense

    def init_eps_and_delta(self, n_features):
        """Implements a heuristic found by [1] to correctly
        set epsilon, and initializes delta with an isotropic
        Gaussian distribution.

        Parameters
        ----------
        n_features: int
            Number of features in the design matrix.
        """
        self.eps = 2 * self.sigma / (n_features ** 0.3)
        self.delta = self.rng.randn(n_features, n_features)
