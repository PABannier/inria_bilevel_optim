import numpy as np
from sklearn.utils import check_random_state


def simulate_data(
    n_samples=100,
    n_features=1000,
    n_tasks=100,
    nnz=10,
    snr=3,
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

    dense_W = rng.randn(n_features, n_tasks)
    sparse_mat = np.zeros_like(dense_W)

    for i in range(n_tasks):
        col = np.array([1] * nnz + [0] * (n_features - nnz)).T
        rng.shuffle(col)
        sparse_mat[:, i] = col

    W = dense_W * sparse_mat
    Y = X @ W

    # Adding a pre-defined SNR to a signal
    # Source: https://sites.ualberta.ca/~msacchi/SNR_Def.pdf
    noise = rng.randn(n_samples, n_tasks)
    noise_std = np.sqrt(
        (np.linalg.norm(Y, axis=0) ** 2) / (snr * np.linalg.norm(noise, axis=0) ** 2)
    )
    Y += noise_std * noise

    return X, Y, W
