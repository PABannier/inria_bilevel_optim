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


def test_sine_waves():
    rng = check_random_state(0)

    n_samples, n_features, n_tasks = 20, 100, 50
    n_relevant_features = 30

    support = rng.choice(n_features, n_relevant_features, replace=False)

    coef = np.zeros((n_features, n_tasks))
    times = np.linspace(0, 2 * np.pi, n_tasks)

    for k in support:
        coef[k, :] = np.sin((1.0 + rng.randn(1)) * times + 3 * rng.randn(1))

    X = rng.randn(n_samples, n_features)
    Y = X @ coef + rng.randn(n_samples, n_tasks)
    Y /= norm(Y, ord="fro")

    regressor = ReweightedMTL()
    regressor.fit(X, Y)

    np.testing.assert_allclose(regressor.weights, coef, atol=1e-5)
