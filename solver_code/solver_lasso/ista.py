import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state

from solver_lasso.utils import (
    get_duality_gap,
    ST,
    fista_iteration,
    ista_iteration,
)


class ProxLasso(BaseEstimator, RegressorMixin):
    """LASSO solved using proximal gradient descent
    (ISTA) and a faster variant (FISTA).

    ProxLasso implements two stopping critera:
        - max_iter is reached
        - the duality gap is below some threshold

    Parameters
    ----------
    alpha : float, default=0.1
        Constants that multiplies the regularizer.

    max_iter : iter, default=100
        Maximum number of iterations for (F)ISTA

    dual_threshold : float, default=1e-5
        Dual suboptimality gap

    verbose : bool, default=True
        Option to print the loss when fitting the
        estimator.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)

    loss_history_ : list
        Contains the training loss history after fitting.

    dual_gap_ : float
        The duality gap between primal and dual problems.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        verbose: bool = True,
        accelerated: bool = True,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.accelerated = accelerated

        self.coef_ = None
        self.loss_history_ = []
        self.gap_history_ = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits estimator to the data.

        Parameters
        ----------
        X: np.ndarray of shape (n_samples, n_features)
            Design matrix.

        y: np.ndarray of shape (n_features,)
            Target vector.
        """
        X, y = check_X_y(X, y)
        L = norm(X, ord=2) ** 2

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        if self.accelerated:
            z = self.coef_.copy()
            t = 1

        for iter_idx in range(self.max_iter):
            if self.accelerated:
                self.coef_, t, z = fista_iteration(
                    self.coef_, X, y, t, z, L, self.alpha
                )
            else:
                self.coef_ = ista_iteration(self.coef_, X, y, L, self.alpha)

            gap, p_obj, d_obj = get_duality_gap(X, y, self.coef_, self.alpha)

            self.loss_history_.append(p_obj)
            self.gap_history_.append(gap)

            if gap < self.tol:
                if self.verbose:
                    print("\n")
                    print(
                        f"Fitting ended at iteration {iter_idx} with duality gap {gap}."
                    )
                break

            if self.verbose == True:
                print(
                    f"[{iter_idx}/{self.max_iter}] Primal: {p_obj:.4f}, Dual: {d_obj:.4f}, Duality gap: {gap:.7f}"
                )

    def predict(self, X: np.ndarray):
        """Predicts new data using fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test matrix.
        """
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.coef_
