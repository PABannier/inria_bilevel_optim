import time

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max


if __name__ == "__main__":
    print("=" * 10)
    print("SINGLE FIT")
    print("=" * 10)
    X, Y, _, _ = simulate_data(
        n_samples=100,
        n_features=150,
        n_tasks=50,
        nnz=5,
        snr=2,
        corr=0.2,
        random_state=0,
    )

    alpha_max = compute_alpha_max(X, Y)

    regressor1 = ReweightedMultiTaskLasso(
        alpha_max / 10, warm_start=True, verbose=False
    )
    regressor2 = ReweightedMultiTaskLasso(
        alpha_max / 10, warm_start=False, verbose=False
    )

    start1 = time.time()
    regressor1.fit(X, Y)
    duration1 = time.time() - start1

    start2 = time.time()
    regressor2.fit(X, Y)
    duration2 = time.time() - start2

    print("Warm start=True", duration1)
    print("Warm start=False", duration2)

    print("\n")

    print("=" * 10)
    print("CV")
    print("=" * 10)

    alphas = np.geomspace(alpha_max, alpha_max / 20, 30)

    criterion1 = ReweightedMultiTaskLassoCV(
        alphas, n_iterations=10, warm_start=True
    )
    criterion2 = ReweightedMultiTaskLassoCV(
        alphas, n_iterations=10, warm_start=False
    )

    start1 = time.time()
    criterion1.fit(X, Y)
    duration1 = time.time() - start1

    start2 = time.time()
    criterion2.fit(X, Y)
    duration2 = time.time() - start2

    print("\n")

    print("Warm start=True", duration1)
    print("Warm start=False", duration2)

    print("\n")
