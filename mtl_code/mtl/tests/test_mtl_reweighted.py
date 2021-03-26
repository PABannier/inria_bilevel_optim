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
