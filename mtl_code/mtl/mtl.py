import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state

from celer import MultiTaskLasso
from mtl.solver_free_orient import MultiTaskLassoOrientation


class ReweightedMultiTaskLasso(BaseEstimator, RegressorMixin):
    """Reweighted Multi-Task LASSO.

    Parameters
    ----------
    alpha : float, default=0.1
        Constants that multiplies the L1/L2 mixed norm as a regularizer.

    n_iterations : int
        Number of reweighting iterations performed during fitting.

    verbose : bool, default=True
        Option to print the loss when fitting the estimator.

    penalty : callable, default=None
        Custom penalty to rescale the weights after one iteration.
        By default, the penalty is the same as in [1].

    tol : float, default=1e-4
        Duality gap tolerance for MultiTaskLASSO solver.

    warm_start : bool, default=True
        Enables MultiTaskLasso to start from the previous fit if available.
        It might cause some issue if the right version of Celer is not installed
        in the system. If you have any issues, set it to False to solve the issue.

    n_orient : int, default=1
        Number of orientations for a dipole on the scalp surface. Choose 1 for fixed
        orientation and 3 for free orientation.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features, n_tasks)
        Parameter matrix of coefficients for the Multi-Task LASSO.

    loss_history_ : list
        Contains the training loss history after fitting.

    References
    ----------
    .. [1] Candès et al. (2007), Enhancing sparsity by reweighted l1 minimization
           https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf
    """

    def __init__(
        self,
        alpha: float = 0.1,
        n_iterations: int = 10,
        verbose: bool = True,
        penalty: callable = None,
        tol: float = 1e-4,
        warm_start: bool = True,
        n_orient: int = 1,
    ):
        self.alpha = alpha
        self.verbose = verbose
        self.n_iterations = n_iterations
        self.warm_start = warm_start
        self.tol = tol

        self.n_orient = n_orient

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1 / (
                2 * np.sqrt(np.linalg.norm(u, axis=1)) + np.finfo(float).eps
            )

        self.coef_ = None
        self.loss_history_ = []

        if self.n_orient == 1:
            self.regressor = MultiTaskLasso(
                alpha=alpha,
                fit_intercept=False,
                warm_start=self.warm_start,
                tol=self.tol,
            )
            self.transpose = True  # Celer output weights (n_times, n_features)
        elif self.n_orient > 1:
            self.regressor = MultiTaskLassoOrientation(
                alpha=alpha,
                n_orient=self.n_orient,
                tol=self.tol,
                warm_start=self.warm_start,
            )
            self.transpose = False
        else:
            raise ValueError(
                "Number of orientations must be strictly positive. "
                + "Hint: 1 for fixed orientation, 3 for free orientation."
            )

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fits estimator to the data.

        Training consists in fitting multiple Multi-Task LASSO estimators
        by iteratively reweighting the coefficient matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y : np.ndarray of shape (n_samples, n_tasks)
            Target matrix.
        """
        X, Y = check_X_y(X, Y, multi_output=True)
        n_samples, n_features = X.shape

        w = np.ones(n_features)

        objective = lambda W: np.sum((Y - X @ W) ** 2) / (
            2 * n_samples
        ) + self.alpha * np.sum(np.sqrt(norm(W, axis=1)))

        for l in range(self.n_iterations):
            # Trick: rescaling the weights
            X_w = X / w[np.newaxis, :]

            # Solving weighted l1 minimization problem
            self.regressor.fit(X_w, Y)

            # Trick: "de-scaling" the weights
            if self.transpose:
                coef_hat = (self.regressor.coef_ / w).T
            else:
                coef_hat = (self.regressor.coef_.T / w).T

            # Updating the weights
            w = self.penalty(coef_hat)

            loss = objective(coef_hat)
            self.loss_history_.append(loss)

            if self.verbose:
                print(f"Iteration {l}: {loss:.4f}")

        self.coef_ = coef_hat

    def predict(self, X: np.ndarray):
        """Predicts data with the fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix for inference.
        """
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.coef_
