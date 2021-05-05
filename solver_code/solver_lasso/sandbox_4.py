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
    get_duality_gap_mtl_as,
    primal_mtl_as,
    dual_mtl_as,
    groups_norm2,
    norm_l2_inf,
)

from solver_free_orient import dgap_l21
import ipdb

import time

if __name__ == "__main__":
    MAX_ITER = 2000
    TOL = 1e-8
    K = 5
    N_ORIENT = 3

    ACTIVE_SET_SIZE = 10

    use_acc = True

    gap_history_ = []

    X, Y, W, _ = simulate_data(
        n_samples=100, n_features=999, n_tasks=200, nnz=10, random_state=0
    )

    alpha_max = norm_l2_inf(np.dot(X.T, Y), N_ORIENT, copy=False)
    alpha = alpha_max * 0.5

    start = time.time()

    n_samples, n_features = X.shape
    n_times = Y.shape[1]

    n_positions = n_features // N_ORIENT

    L = np.empty(n_positions)

    for j in range(n_positions):
        idx = slice(j * N_ORIENT, (j + 1) * N_ORIENT)
        L[j] = norm(X[:, idx], ord=2) ** 2

    active_set = np.zeros(n_features, dtype=bool)
    idx_large_corr = np.argsort(groups_norm2(np.dot(X.T, Y), N_ORIENT))
    new_active_idx = idx_large_corr[-ACTIVE_SET_SIZE:]

    if N_ORIENT > 1:
        new_active_idx = (
            N_ORIENT * new_active_idx[:, None] + np.arange(N_ORIENT)[None, :]
        ).ravel()

    active_set[new_active_idx] = True
    as_size = np.sum(active_set)

    coef_init = None

    # ============== MIXED NORM SOLVER BCD ==============

    for k in range(MAX_ITER):

        L_tmp = L[active_set[::N_ORIENT]]

        def solver(X, Y, alpha, L, init=None):

            n_samples, n_times = Y.shape
            n_samples, n_features = X.shape
            n_positions = n_features // N_ORIENT

            if init is None:
                coef = np.zeros((n_features, n_times))
                R = Y.copy()
            else:
                coef = init
                R = Y - X @ coef

            X = np.asfortranarray(X)
            Y = np.asfortranarray(Y)

            if use_acc:
                last_K_coef = np.empty((K + 1, n_features, n_times))
                U = np.zeros((K, n_features * n_times))

            active_set = np.zeros(n_features, dtype=bool)

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
                        active_set[idx] = False
                    else:
                        shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                        coef_j_new *= shrink

                        R -= np.dot(X_j, coef_j_new)
                        coef_j[:] = coef_j_new
                        active_set[idx] = True

                gap, p_obj, d_obj = get_duality_gap_mtl_as(
                    X, Y, coef[active_set], active_set, alpha, N_ORIENT
                )
                gap_history_.append(gap)

                print(
                    f"[{iter_idx+1}/{MAX_ITER}] p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} :: d_gap {gap:.5f}"
                )

                if use_acc:
                    last_K_coef[iter_idx % (K + 1)] = coef

                    if iter_idx % (K + 1) == K:
                        for k in range(K):
                            U[k] = (
                                last_K_coef[k + 1].ravel()
                                - last_K_coef[k].ravel()
                            )

                        C = U @ U.T

                        try:
                            z = np.linalg.solve(C, np.ones(K))
                            c = z / z.sum()

                            coef_acc = np.sum(
                                last_K_coef[:-1] * c[:, None, None], axis=0
                            )

                            active_set_acc = norm(coef_acc, axis=1) != 0

                            p_obj_acc = primal_mtl_as(
                                X,
                                Y,
                                coef_acc[active_set_acc],
                                active_set_acc,
                                alpha,
                                N_ORIENT,
                            )

                            # ipdb.set_trace()

                            if p_obj_acc < p_obj:
                                print("Extrapolation worked!")
                                coef = coef_acc
                                active_set = active_set_acc
                                R = Y - X[:, active_set] @ coef[active_set]
                            else:
                                print(
                                    f"p_obj {p_obj} :: p_obj_acc {p_obj_acc}"
                                )

                        except np.linalg.LinAlgError:
                            print("LinAlg Error")

                if gap < TOL:
                    print(f"Fitting ended after iteration {iter_idx + 1}.")
                    break

            coef = coef[active_set]
            return coef, active_set

        coef, as_ = solver(X[:, active_set], Y, alpha, L_tmp, coef_init)

        active_set[active_set] = as_.copy()  # ?????????
        idx_old_active_set = np.where(active_set)[0]

        gap, p_obj, d_obj = get_duality_gap_mtl_as(
            X, Y, coef, active_set, alpha, N_ORIENT
        )

        gap_history_.append(gap)

        print(
            f"[{k+1}/{MAX_ITER}] p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} :: d_gap {gap:.5f} :: n_active_start {as_size // N_ORIENT} :: n_active_end {np.sum(active_set) // N_ORIENT}"
        )

        if gap < TOL:
            print("Convergence reached!")
            break

        if k < (MAX_ITER - 1):
            R = Y - X[:, active_set] @ coef
            idx_large_corr = np.argsort(groups_norm2(np.dot(X.T, R), N_ORIENT))
            new_active_idx = idx_large_corr[-ACTIVE_SET_SIZE:]

            if N_ORIENT > 1:
                new_active_idx = (
                    N_ORIENT * new_active_idx[:, None]
                    + np.arange(N_ORIENT)[None, :]
                )
                new_active_idx = new_active_idx.ravel()

            active_set[new_active_idx] = True
            idx_active_set = np.where(active_set)[0]
            as_size = np.sum(active_set)
            coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
            idx = np.searchsorted(idx_active_set, idx_old_active_set)
            coef_init[idx] = coef

    print(f"Duration: {time.time() - start}")

    fig = plt.figure()

    plt.plot(np.log10(gap_history_), label="Gap")

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("$p^* - d^*$ (logarithmic)")

    plt.title("Convergence speed", fontsize=13)

    fig.show()
