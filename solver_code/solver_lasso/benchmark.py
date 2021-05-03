import time

from mne.inverse_sparse.mxne_optim import _mixed_norm_solver_bcd

from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max

from solver_free_orient import MultiTaskLasso, compute_lipschitz_constants


def time_solver(X, Y, n_orient):
    alpha_max = compute_alpha_max(X, Y)
    alpha = alpha_max * 0.2

    estimator1 = MultiTaskLasso(
        n_orient=n_orient, max_iter=10000, verbose=True
    )

    start = time.time()
    estimator1.fit(X, Y, alpha)
    duration1 = time.time() - start

    lc = compute_lipschitz_constants(X, X.shape[1] // n_orient, n_orient)
    start = time.time()
    _mixed_norm_solver_bcd(
        Y, X, alpha, lc, maxit=10000, tol=1e-5, n_orient=n_orient
    )
    duration2 = time.time() - start

    return duration1, duration2


if __name__ == "__main__":
    X, Y, W, _ = simulate_data(
        n_samples=100, n_features=522, n_tasks=50, random_state=0
    )

    duration1_fixed, duration2_fixed = time_solver(X, Y, 1)
    duration1_free, duration2_free = time_solver(X, Y, 3)

    print("==== FIXED ====")
    print("PA:", duration1_fixed)
    print("MNE:", duration2_fixed)
    print("\n")

    print("==== FREE =====")
    print("PA:", duration1_free)
    print("MNE:", duration2_free)
