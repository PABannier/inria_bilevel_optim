import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, f1_score, jaccard_score
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
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
    alphas : list or np.ndarray
        Values of `alpha` to test.

    criterion : Callable, default=mean_squared_error
        Cross-validation metric (e.g. MSE, SURE).

    n_folds : int, default=5
        Number of folds.

    n_iterations : int
        Number of reweighting iterations performed during fitting.

    random_state : int or None
        Seed for reproducible experiments.
    """

    def __init__(
        self,
        alphas: list,
        criterion=mean_squared_error,
        n_folds: int = 5,
        n_iterations: int = 5,
        random_state: int = None,
    ):
        if not isinstance(alphas, (list, np.ndarray)):
            raise TypeError(
                "The parameter grid must be a list or a Numpy array."
            )

        self.alphas = alphas
        self.criterion = criterion
        self.n_folds = n_folds
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.best_estimator_ = None
        self.best_cv_, self.best_alpha_ = np.inf, None

        self.mse_path_ = np.zeros((len(alphas), n_folds))
        self.f1_path_ = np.zeros((len(alphas), n_folds))
        self.jaccard_path_ = np.zeros((len(alphas), n_folds))

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

        if X.shape[0] < self.n_folds:
            raise ValueError(
                "The number of folds can't be greater than the number of samples."
            )

        kf = KFold(self.n_folds, random_state=self.random_state)

        for idx_alpha, alpha_param in enumerate(self.alphas):
            print("Fitting MTL estimator with alpha =", alpha_param)
            estimator_ = ReweightedMultiTaskLasso(
                alpha_param, n_iterations=self.n_iterations, verbose=False
            )

            Y_oof = np.zeros_like(Y)

            for idx_fold, (train_indices, valid_indices) in enumerate(
                kf.split(X, Y)
            ):
                X_train, Y_train = X[train_indices, :], Y[train_indices, :]
                X_valid, Y_valid = X[valid_indices, :], Y[valid_indices, :]

                estimator_.fit(X_train, Y_train)
                Y_pred = estimator_.predict(X_valid)
                Y_oof[valid_indices, :] = Y_pred

                self.mse_path_[idx_alpha, idx_fold] = mean_squared_error(
                    Y_valid, Y_pred
                )

                if coef_true is not None:
                    self.f1_path_[idx_alpha, idx_fold] = f1_score(
                        coef_true != 0, estimator_.coef_ != 0, average="macro"
                    )

                    self.jaccard_path_[idx_alpha, idx_fold] = jaccard_score(
                        coef_true != 0, estimator_.coef_ != 0, average="macro"
                    )

            cv_score = self.criterion(Y, Y_oof)

            if cv_score < self.best_cv_:
                print(
                    f"Criterion reduced from {self.best_cv_:.5f} to "
                    + f"{cv_score:.5f} for alpha = {alpha_param}"
                )

                self.best_cv_ = cv_score
                self.best_alpha_ = alpha_param
                self.best_estimator_ = estimator_

        print("\n")
        print(f"Best criterion: {self.best_cv_}")
        print(f"Best alpha: {self.best_alpha_}")

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
