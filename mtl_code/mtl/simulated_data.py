import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state


def simulate_data(
    n_samples=100,
    n_features=1000,
    n_tasks=150,
    nnz=10,
    snr=1,
    random_state=None,
):
    """Generates a simulated dataset and a row-sparse weight
       matrix for multi-task lasso.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
        Corresponds to the number of sensors on the scalp.

    n_features : int, default=1000
        Number of features.
        Corresponds to the discretized areas in the brain.

    n_tasks : int, default=100
        Number of tasks.
        Corresponds to the time series length of the brain recording.

    nnz : int, default=10
        Number of non-zero coefficients.

    snr : float, default=3
        Signal-to-noise ratio.

    random_state : int, default=None
        Seed for random number generators.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix.

    Y : np.ndarray of shape (n_samples, n_tasks)
        Target matrix.

    W : np.ndarray of shape (n_features, n_tasks)
        Sparse-row weight matrix.
    """

    rng = check_random_state(random_state)

    if nnz > n_features:
        raise ValueError(
            "Number of non-zero coefficients can't be greater than the number of features"
        )

    X = rng.randn(n_samples, n_features)

    support = rng.choice(n_features, nnz, replace=False)
    W = np.zeros((n_features, n_tasks))

    for k in support:
        W[k, :] = rng.normal(size=(n_tasks))

    Y = X @ W + rng.normal(0, snr, (n_samples, n_tasks))
    Y /= norm(Y, ord="fro")

    return X, Y, W
