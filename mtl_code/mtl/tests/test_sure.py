import time
from collections import defaultdict

import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state, check_X_y
from celer import MultiTaskLasso

from mtl.sure import SURE
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.mtl import ReweightedMultiTaskLasso

from mtl.utils_datasets import compute_alpha_max


def test_warm_start():
    random_state = 42

    N_SAMPLES = 102
    N_FEATURES = 1000
    N_TASKS = 100
    NNZ = 4

    # N_SAMPLES = 10
    # N_FEATURES = 20
    # N_TASKS = 2
    # NNZ = 2

    n_alphas = 20
    n_iterations = 5

    X, Y, _, sigma = simulate_data(
        N_SAMPLES, N_FEATURES, N_TASKS, NNZ, random_state=random_state
    )

    max_alpha = compute_alpha_max(X, Y)
    print("alpha max", max_alpha)
    alphas = np.geomspace(max_alpha, max_alpha / 30, n_alphas)

    start_time = time.time()
    criterion2 = SUREForReweightedMultiTaskLasso(
        sigma, alphas, n_iterations, random_state=random_state
    )
    best_sure_, best_alpha_ = criterion2.get_val(X, Y)
    print(f"Best SURE: {best_sure_}")
    print(f"Best alpha: {best_alpha_}")
    print("Duration (with warm start):", time.time() - start_time)

    criterion = SURE(
        ReweightedMultiTaskLasso, sigma, random_state=random_state
    )
    start_time = time.time()
    best_sure, best_alpha = np.inf, None
    for alpha in alphas:
        sure_val = criterion.get_val(
            X,
            Y,
            alpha,
            warm_start=False,
            verbose=False,
            n_iterations=n_iterations,
        )
        if sure_val < best_sure:
            best_sure = sure_val
            best_alpha = alpha

    print("-" * 80)
    print(f"Best SURE: {best_sure}")
    print(f"Best alpha: {best_alpha}")
    print("Duration (without warm start):", time.time() - start_time)

    assert best_alpha == best_alpha_
    np.testing.assert_almost_equal(best_sure_, best_sure)
