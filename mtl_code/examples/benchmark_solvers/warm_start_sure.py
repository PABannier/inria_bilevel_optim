import time

import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state

from mtl.simulated_data import simulate_data
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max


if __name__ == "__main__":
    rs = np.random.RandomState(0)

    X, Y, _, sigma = simulate_data(
        n_samples=100,
        n_features=150,
        n_tasks=50,
        nnz=5,
        snr=2,
        corr=0.2,
        random_state=rs,
    )

    alpha_max = compute_alpha_max(X, Y)
    alphas = np.geomspace(alpha_max, alpha_max / 20, 50)

    criterion = SURE(ReweightedMultiTaskLasso, sigma, random_state=rs)

    start = time.time()
    for alpha in alphas:
        criterion.get_val(X, Y, alpha)
    print(f"Warm start = True: {time.time() - start}")

    print("\n")
    start = time.time()
    for alpha in alphas:
        criterion.get_val(X, Y, alpha, warm_start=False)
    print(f"Warm start = False: {time.time() - start}")
