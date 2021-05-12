import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max

from solver_lasso.utils import (
    soft_thresh,
    block_soft_thresh,
    get_duality_gap_mtl,
    primal,
    anderson_extrapolation,
    cd_iteration,
    sum_squared,
    primal_mtl,
    norm_l2_inf,
)

import ipdb

import time


if __name__ == "__main__":
    MAX_ITER = 2000
    TOL = 1e-8
    K = 5
    N_ORIENT = 3

    use_acc = True

    gap_history_ = []

    X, Y, W, _ = simulate_data(
        n_samples=100, n_features=999, n_tasks=200, nnz=10, random_state=0
    )

    R = Y.copy()

    alpha_max = norm_l2_inf(np.dot(X.T, Y), N_ORIENT, copy=False)
    alpha = alpha_max * 0.1

    start = time.time()

    n_samples, n_features = X.shape
    n_times = Y.shape[1]

    n_positions = n_features // N_ORIENT

    coef = np.zeros((n_features, n_times))

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)

    if use_acc:
        last_K_coef = np.empty((K + 1, n_features, n_times))
        U = np.zeros((K, n_features * n_times))

    L = np.empty(n_positions)

    for j in range(n_positions):
        idx = slice(j * N_ORIENT, (j + 1) * N_ORIENT)
        L[j] = norm(X[:, idx], ord=2) ** 2

    for iter_idx in range(MAX_ITER):
        for j in range(n_positions):
            idx = slice(j * N_ORIENT, (j + 1) * N_ORIENT)
            coef_j = coef[idx]
            X_j = X[:, idx]

            coef_j_new = X_j.T @ R / L[j]

            if coef_j[0, 0] != 0:
                R += X_j @ coef_j
                coef_j_new += coef_j

            block_norm = np.sqrt(sum_squared(coef_j_new))

            alpha_lc = alpha / L[j]

            if block_norm <= alpha_lc:
                coef_j.fill(0.0)
            else:
                shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                coef_j_new *= shrink

                R -= np.dot(X_j, coef_j_new)
                coef_j[:] = coef_j_new

        gap, p_obj, d_obj = get_duality_gap_mtl(X, Y, coef, alpha, N_ORIENT)
        gap_history_.append(gap)

        print(
            f"[{iter_idx+1}/{MAX_ITER}] p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} :: d_gap {gap:.5f}"
        )

        if use_acc:
            last_K_coef[iter_idx % (K + 1)] = coef

            if iter_idx % (K + 1) == K:
                for k in range(K):
                    U[k] = last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()

                C = U @ U.T

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()

                    coef_acc = np.sum(
                        last_K_coef[:-1] * c[:, None, None], axis=0
                    )

                    p_obj_acc = primal_mtl(X, Y, coef_acc, alpha, N_ORIENT)

                    # ipdb.set_trace()

                    if p_obj_acc < p_obj:
                        print("Extrapolation worked!")
                        coef = coef_acc
                        R = Y - X @ coef
                    else:
                        print(f"p_obj {p_obj} :: p_obj_acc {p_obj_acc}")

                except np.linalg.LinAlgError:
                    print("LinAlg Error")

        if gap < TOL:
            print(f"Fitting ended after iteration {iter_idx + 1}.")
            break

    print(f"Duration: {time.time() - start}")

    fig = plt.figure()

    plt.plot(np.log(gap_history_), label="Gap")

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("$p^* - d^*$ (logarithmic)")

    plt.title("Convergence speed", fontsize=13)

    fig.show()
