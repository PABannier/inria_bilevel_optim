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
    X, Y, W = simulate_data(50, 250, 25, 25, random_state=0)

    regressor = ReweightedMTL(alpha=0.0009)
    regressor.fit(X, Y, n_iterations=10)

    nnz_original = np.count_nonzero(np.count_nonzero(W, axis=1))
    nnz_reconstructed = np.count_nonzero(np.count_nonzero(regressor.weights, axis=1))

    assert np.abs(nnz_original - nnz_reconstructed) <= 1
