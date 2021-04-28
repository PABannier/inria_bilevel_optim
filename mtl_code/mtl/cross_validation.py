from collections import defaultdict

import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from celer import MultiTaskLasso

from mtl.mtl import ReweightedMultiTaskLasso


class ReweightedMultiTaskLassoCV(BaseEstimator, RegressorMixin):
    """Cross-validate the regularization penalty constant `alpha`
    for a reweighted multi-task LASSO regression.

    In an inverse problem in neuroscience, partitioning X into
    folds consists in partitioning with respect to the sensors
    on the scalp. This is why CV makes less sense in this kind
    of inverse problem than on vanilla prediction problems. In
    vanilla prediction problems, samples in X are expected to
    be i.i.d., while in an inverse problem like this one X
    represents the geometry of the brain and data fails to be
    i.i.d.

    Parameters
    ----------
    alpha_grid : list or np.ndarray
        Values of `alpha` to test.

    criterion : Callable, default=mean_squared_error
        Cross-validation metric (e.g. MSE, SURE).

    n_folds : int, default=5
        Number of folds.

    n_iterations : int, default=5
        Number of reweighting iterations performed during fitting.

    random_state : int or None, default=None
        Seed for reproducible experiments.

    penalty : callable, default=None
        See docs of ReweightedMultiTaskLasso for more details.
    """

    def __init__(
        self,
        alpha_grid: list,
        criterion: callable = mean_squared_error,
        n_folds: int = 5,
        n_iterations: int = 5,
        random_state: int = None,
        penalty: callable = None,
    ):
        if not isinstance(alpha_grid, (list, np.ndarray)):
            raise TypeError(
                "The parameter grid must be a list or a Numpy array."
            )

        self.alpha_grid = alpha_grid
        self.criterion = criterion
        self.n_folds = n_folds
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.best_estimator_ = None
        self.best_cv_, self.best_alpha_ = np.inf, None

        self.mse_path_ = np.zeros((len(alpha_grid), n_folds))
        self.f1_path_ = np.zeros((len(alpha_grid), n_folds))
        self.jaccard_path_ = np.zeros((len(alpha_grid), n_folds))

        self.n_alphas = len(self.alpha_grid)

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1 / (
                2 * np.sqrt(norm(u, axis=1)) + np.finfo(float).eps
            )

    @property
    def coef_(self):
        return self.best_estimator_.coef_

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        coef_true: np.ndarray = None,
    ):
        """Fits the cross-validation error estimator
        on X and Y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y : np.ndarray of shape (n_samples, n_tasks)
            Target matrix.

        coef_true : np.ndarray of shape (n_features, n_tasks)
            Coefficient matrix. To compute f1_score and jaccard_score paths,
            it needs to be specified.
        """
        X, Y = check_X_y(X, Y, multi_output=True)

        n_samples = X.shape[0]
        n_tasks = Y.shape[1]

        scores_per_alpha_ = [np.inf for _ in range(self.n_alphas)]
        Y_oofs_ = [
            np.zeros((n_samples, n_tasks)) for _ in range(self.n_alphas)
        ]

        kf = KFold(self.n_folds, random_state=self.random_state, shuffle=True)

        for i, (trn_idx, val_idx) in enumerate(kf.split(X, Y)):
            print(f"Fitting fold {i+1}...")
            X_train, Y_train = X[trn_idx, :], Y[trn_idx, :]
            X_valid, Y_valid = X[val_idx, :], Y[val_idx, :]

            coefs_ = self._fit_reweighted_with_grid(
                X_train, Y_train, X_valid, Y_valid, coef_true, i
            )

            predictions_ = [X_valid @ coefs_[j] for j in range(self.n_alphas)]

            for i in range(len(Y_oofs_)):
                Y_oofs_[i][val_idx, :] = predictions_[i]

        for i in range(len(Y_oofs_)):
            scores_per_alpha_[i] = self.criterion(Y, Y_oofs_[i])

        self.best_cv_ = np.min(scores_per_alpha_)
        self.best_alpha_ = self.alpha_grid[np.argmin(scores_per_alpha_)]

        print("Refitting with best alpha...")
        self.best_estimator_ = ReweightedMultiTaskLasso(
            self.best_alpha_, penalty=self.penalty, verbose=False
        )

        self.best_estimator_.fit(X, Y)

        print("\n")
        print(f"Best criterion: {self.best_cv_}")
        print(f"Best alpha: {self.best_alpha_}")

    def _fit_reweighted_with_grid(
        self, X_train, Y_train, X_valid, Y_valid, coef_true, idx_fold
    ):
        n_features = X_train.shape[1]
        n_tasks = Y_train.shape[1]

        coef_0 = np.empty((self.n_alphas, n_features, n_tasks))

        regressor = MultiTaskLasso(
            np.nan, fit_intercept=False, warm_start=True
        )

        # Copy grid of first iteration (leverages convexity)
        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha
            coef_0[j] = regressor.fit(X_train, Y_train).coef_.T

        regressor.warm_start = False
        coefs = coef_0.copy()

        for j, alpha in enumerate(self.alpha_grid):
            regressor.alpha = alpha

            w = self.penalty(coef_0[j])

            for _ in range(self.n_iterations - 1):
                mask = w != 1.0 / np.finfo(float).eps
                coefs[j][~mask] = 0.0

                if mask.sum():
                    coefs[j][mask], w[mask] = self._reweight_op(
                        regressor, X_train[:, mask], Y_train, w[mask]
                    )

                    self.mse_path_[j, idx_fold] = mean_squared_error(
                        Y_valid, X_valid @ coefs[j]
                    )

                    if coef_true is not None:
                        self.f1_path_[j, idx_fold] = f1_score(
                            coef_true != 0, coefs[j] != 0, average="macro"
                        )

                        self.jaccard_path_[j, idx_fold] = jaccard_score(
                            coef_true != 0, coefs[j] != 0, average="macro"
                        )

        return coefs

    def _reweight_op(self, regressor, X, Y, w):
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, Y)

        coef = (regressor.coef_ / w).T
        w = self.penalty(coef)

        return coef, w

    def predict(self, X: np.ndarray):
        """Predicts data with the fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix for inference.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.best_estimator_.predict(X)
