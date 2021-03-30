import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state


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
    TBD

    References
    ----------
    .. [1] C.-A. Deledalle, Stein Unbiased GrAdient estimator of the Risk
    (SUGAR) for multiple parameter selection.
    SIAM J. Imaging Sci., 7(4), 2448-2487.

    """

    def __init__(self, sigma, random_state=None):
        self.sigma = sigma
        self.rng = check_random_state(random_state)

    def __call__(self, X, Y, mask, dense, mask2, dense2):
        """Computes the SURE approximation.

        Parameters:
        X: np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y: np.ndarray of shape (n_samples, n_tasks)
            Target matrix.

        mask: np.ndarray of shape (n_features, n_tasks)
            Boolean array corresponding to the non-zero
            coefficients of the solution of the inner
            optimization problem.

        dense: np.ndarray
            Values of the non-zero coefficients of the
            solution of the first optimization problem.

        mask2: np.ndarray of shape (n_features, n_tasks)
            Boolean array corresponding to the non-zero
            coefficients of the solution of the outer
            optimization problem.

        dense2: np.ndarray
            Values of the non-zero coefficients of the
            solution of the second optimization problem.
        """
