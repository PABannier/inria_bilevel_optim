from mtl.mtl import ReweightedMTL
from mtl.simulated_data import simulate_data
import numpy as np


def test_training_loss_decrease():
    X, Y, W = simulate_data(n_samples=10, n_features=20, n_tasks=15)

    regressor = ReweightedMTL()
    regressor.fit(X, Y)

    start_loss = regressor.loss_history_[0]
    final_loss = regressor.loss_history_[-1]

    assert start_loss > final_loss


def test_predict():
    X_tr, Y_tr, W = simulate_data(n_samples=10, n_features=20, n_tasks=15)
    X_test, _, _ = simulate_data(n_samples=40, n_features=20, n_tasks=15)

    regressor = ReweightedMTL()
    regressor.fit(X_tr, Y_tr)

    regressor.predict(X_test)
