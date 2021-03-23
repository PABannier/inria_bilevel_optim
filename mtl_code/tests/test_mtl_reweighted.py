from mtl_code.mtl import ReweightedMTL
from mtl_code.simulated_data import simulate_data
import numpy as np


def test_training_loss_decrease():
    X, Y, W = simulate_data()

    regressor = ReweightedMTL(verbose=True)
    regressor.fit(X, Y)

    assert False
