from mtl.mtl import ReweightedMTL
from mtl.simulated_data import simulate_data
import numpy as np


def test_training_loss_decrease():
    X, Y, W = simulate_data()

    regressor = ReweightedMTL(verbose=True)
    regressor.fit(X, Y, n_iterations=20)

    assert False