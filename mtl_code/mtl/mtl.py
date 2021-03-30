import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state
from celer import MultiTaskLasso
# from sklearn.linear_model import MultiTaskLasso


class ReweightedMTL(BaseEstimator, RegressorMixin):
    """Reweighted Multi-Task LASSO.

    Parameters
    ----------
    alpha : float, default=0.1
        Constants that multiplies the L1/L2 mixed norm as a regularizer.

    verbose : bool, default=True
        Option to print the loss when fitting the estimator.

    Attributes
    ----------
    weights : np.ndarray of shape (n_features, n_tasks)
        Parameter matrix of coefficients for the Multi-Task LASSO.

    loss_history : list
        Contains the training loss history after fitting.

    n_iterations : int
        Number of reweighting iterations performed during fitting.

    References
    ----------
    .. [1] Candès et al. (2007), Enhancing sparsity by reweighted l1 minimization
           https://web.stanford.edu/~boyd/papers/pdf/rwl1.pdf
    """

    def __init__(
        self, alpha: float = 0.1, n_iterations: int = 10, verbose: bool = True
    ):
        self.alpha = alpha
        self.verbose = verbose
        self.n_iterations = n_iterations

        self.weights = None
        self.loss_history_ = []
        self.clf = MultiTaskLasso(alpha=alpha, fit_intercept=False, warm_start=True)

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
        n_samples, n_features = X.shape[0], X.shape[1]

        w = np.ones(n_features)

        objective = lambda W: np.sum((Y - X @ W) ** 2) / (
            2 * n_samples
        ) + self.alpha * np.sum(np.sqrt(norm(W, axis=1)))

        for l in range(self.n_iterations):
            # Trick: rescaling the weights
            X_w = X / w[np.newaxis, :]

            # Solving weighted l1 minimization problem
            self.clf.fit(X_w, Y)

            # Trick: "de-scaling" the weights
            coef_hat = (self.clf.coef_ / w).T  # (n_features, n_tasks)

            # Updating the weights
            c = np.linalg.norm(coef_hat, axis=1)
            w = 1 / (2 * np.sqrt(c) + np.finfo(float).eps)

            loss = objective(coef_hat)
            self.loss_history_.append(loss)

            if self.verbose:
                print(f"Iteration {l}: {loss:.4f}")

        self.weights = coef_hat

    def predict(self, X: np.ndarray):
        """Predicts data with the fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix for inference.
        """
        check_is_fitted(self)
        X = check_array(X)

        return X @ self.weights
