import numpy as np


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
