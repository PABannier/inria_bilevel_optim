import numpy as np
from numpy.linalg import norm
from scipy import signal
from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state


def simulate_data(n_samples=100, n_features=1000, n_tasks=150, nnz=10, snr=4, corr=0.3,
                  random_state=None, autocorrelated=False):
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

    corr : float, default=0.3
        Correlation coefficient.

    random_state : int, default=None
        Seed for random number generators.

    autocorrelated : bool, default=False
        If False, the noise is white. If True, it is
        temporally auto-correlated.

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

    if not 0 <= corr < 1:
        raise ValueError("The correlation coefficient must be in [0, 1)")

    if nnz > n_features:
        raise ValueError(
            "Number of non-zero coefficients can't be greater than the number of features"
        )

    sigma = np.sqrt(1 - corr ** 2)
    U = rng.randn(n_samples)

    X = np.empty([n_samples, n_features], order="F")
    X[:, 0] = U
    for j in range(1, n_features):
        U *= corr
        U += sigma * rng.randn(n_samples)
        X[:, j] = U

    support = rng.choice(n_features, nnz, replace=False)
    W = np.zeros((n_features, n_tasks))

    for k in support:
        W[k, :] = rng.normal(size=(n_tasks))

    Y = X @ W

    if autocorrelated:
        noise = rng.randn(n_samples, n_tasks)
        noise_corr = signal.lfilter([1], [1, -0.9], noise, axis=1)
        sigma = 1 / norm(noise_corr) * norm(Y) / snr
        Y += sigma * noise_corr

    else:
        noise = rng.randn(n_samples, n_tasks)
        sigma = 1 / norm(noise) * norm(Y) / snr

        Y += sigma * noise

    return X, Y, W, sigma


if __name__ == "__main__":
    n_samples = 100
    n_features = 1000
    n_tasks = 1

    X, y = simulate_data(n_samples, n_features, n_tasks, nnz=3, random_state=0)[:2]
    y = np.ravel(y)
    alpha_max = norm(X.T @ y, ord=np.inf) / len(X)
    alpha = alpha_max * 0.01 

    L = norm(X.T @ X, ord=2)

    # Closed-form solution: dof = support size
    clf = Lasso(alpha, fit_intercept=False)
    clf.fit(X, y)
    dof_cf = np.sum(clf.coef_ != 0)

    # Analytical solution via implicit differentiation
    beta = clf.coef_
    z = beta - (1 / L) * X.T @ (X @ beta - y)
    grad_prox_z = np.zeros_like(z)
    grad_prox_z[np.abs(z) >= alpha] = 1
    grad_prox_z = np.diag(grad_prox_z)

    id_feat = np.eye(n_features)
    J = np.linalg.inv(id_feat - grad_prox_z @ (id_feat - X.T @ X / L)) @ grad_prox_z @ X.T / L
    dof_id = np.trace(X @ J)

    print("DOF")
    print("Closed form:", dof_cf)
    print("Implicit differentiation:", dof_id)
