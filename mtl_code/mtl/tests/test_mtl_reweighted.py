from mtl.mtl import ReweightedMTL
from mtl.simulated_data import simulate_data
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state


def test_training_loss_decrease():
    X, Y, W = simulate_data(n_samples=10, n_features=20, n_tasks=15, nnz=5)

    regressor = ReweightedMTL()
    regressor.fit(X, Y)

    start_loss = regressor.loss_history_[0]
    final_loss = regressor.loss_history_[-1]

    assert start_loss > final_loss


def test_sparsity_level():
    """Tests that the sparsity level is (nearly) identical between
    the original and the reconstructed coefficient matrix.

    The correct hyperparameters have been found in the notebook
    `Sparse reconstructions signal`, available in the example
    folder.
    """
    rng = check_random_state(0)

    n_samples, n_features, n_tasks = 50, 250, 25
    n_relevant_features = 25

    support = rng.choice(n_features, n_relevant_features, replace=False)

    coef = np.zeros((n_features, n_tasks))
    times = np.linspace(0, 2 * np.pi, n_tasks)

    for k in support:
        coef[k, :] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))

    X = rng.randn(n_samples, n_features)
    Y = X @ coef + rng.randn(n_samples, n_tasks)
    Y /= norm(Y, ord="fro")

    regressor = ReweightedMTL(alpha=0.0009)
    regressor.fit(X, Y, n_iterations=10)

    nnz_original = np.count_nonzero(np.count_nonzero(coef, axis=1))
    nnz_reconstructed = np.count_nonzero(np.count_nonzero(regressor.weights, axis=1))

    assert np.abs(nnz_original - nnz_reconstructed) <= 1
