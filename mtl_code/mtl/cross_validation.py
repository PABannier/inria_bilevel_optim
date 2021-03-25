import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from mtl.mtl import ReweightedMTL


class MultiTaskLassoCV(BaseEstimator, RegressorMixin):
    """Cross-validate the regularization penalty constant `alpha`
    for reweighted MultiTaskLASSO.

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
    param_grid : list or np.ndarray
        Values of `alpha` to test.

    criterion : Callable, default=mean_squared_error
        Cross-validation metric (e.g. MSE, SURE).

    n_folds : int, default=5
        Number of folds.

    random_state : int or None
        Seed for reproducible experiments.
    """

    def __init__(
        self,
        param_grid: list,
        criterion=mean_squared_error,
        n_folds: int = 5,
        random_state: int = None,
    ):
        if not isinstance(param_grid, (list, np.ndarray)):
            raise TypeError("The parameter grid must be a list or a Numpy array.")

        self.param_grid = param_grid
        self.criterion = criterion
        self.n_folds = n_folds
        self.random_state = random_state

        self.best_estimator = None
        self.best_cv, self.best_alpha = np.inf, None

    @property
    def weights(self):
        return self.best_estimator.weights

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fits the cross-validation error estimator
        on X and Y.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix.

        Y : np.ndarray of shape (n_samples, n_tasks)
            Target matrix.
        """
        X, Y = check_X_y(X, Y, multi_output=True)

        if X.shape[0] < self.n_folds:
            raise ValueError(
                "The number of folds can't be greater than the number of samples."
            )

        kf = KFold(self.n_folds, random_state=self.random_state)

        for alpha in self.param_grid:
            print("Fitting MTL estimator with alpha =", alpha)
            estimator_ = ReweightedMTL(alpha, verbose=False)

            Y_oof = np.zeros_like(Y)

            for train_indices, valid_indices in kf.split(X, Y):
                X_train, Y_train = X[train_indices, :], Y[train_indices, :]
                X_valid, Y_valid = X[valid_indices, :], Y[valid_indices, :]

                estimator_.fit(X_train, Y_train, n_iterations=10)
                Y_pred = estimator_.predict(X_valid)
                Y_oof[valid_indices, :] = Y_pred

            cv_score = self.criterion(Y, Y_oof)

            if cv_score < self.best_cv:
                print(
                    f"Criterion reduced from {self.best_cv:.5f} to "
                    + f"{cv_score:.5f} for alpha = {alpha}"
                )

                self.best_cv = cv_score
                self.best_alpha = alpha
                self.best_estimator = estimator_

        print("\n")
        print(f"Best criterion: {self.best_cv}")
        print(f"Best alpha: {self.best_alpha}")

    def predict(self, X: np.ndarray):
        """Predicts data with the fitted coefficients.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Design matrix for inference.
        """
        check_is_fitted(self)
        X = check_array(X)
        return self.best_estimator.predict(X)


def mtl_cross_val(estimator, criterion, X, Y, n_folds=5):
    """Carries out a cross validation to estimate the performance
       of an multi-task LASSO estimator.

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
    estimator : BaseEstimator
        Scikit-learn estimator.

    criterion : Callable
        Cross-validation metric (e.g. SURE).

    X : np.ndarray of shape (n_samples, n_features)
        Design matrix.

    Y : np.ndarray of shape (n_samples, n_tasks)
        Target matrix.

    n_folds : int, default=5
        Number of folds.

    Returns
    -------
    loss : float
        Cross-validation loss.
    """

    Y_oof = np.zeros_like(Y)
    n_samples = X.shape[0]

    folds = np.array_split(range(n_samples), n_folds)

    for i in range(n_folds):
        train_indices = np.concatenate([fold for j, fold in enumerate(folds) if i != j])
        valid_indices = folds[i]

        X_train, Y_train = X[train_indices, :], Y[train_indices, :]
        X_valid, Y_valid = X[valid_indices, :], Y[valid_indices, :]

        estimator.fit(X_train, Y_train)
        Y_pred = estimator.predict(X_valid)

        Y_oof[valid_indices, :] = Y_pred

    return criterion(Y, Y_oof)
