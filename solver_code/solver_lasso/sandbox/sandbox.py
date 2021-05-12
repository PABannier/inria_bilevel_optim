import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max

from solver_lasso.utils import (
    soft_thresh,
    block_soft_thresh,
    get_duality_gap,
    primal,
    anderson_extrapolation,
    cd_iteration,
)


if __name__ == "__main__":
    MAX_ITER = 1000
    TOL = 1e-5
    K = 5

    use_acc = True

    gap_history_ = []

    X, y, W, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=1, nnz=3, random_state=0
    )

    y = y.ravel()

    alpha_max = compute_alpha_max(X, y)
    alpha = alpha_max * 0.1

    n_samples, n_features = X.shape

    coef = np.zeros(n_features)

    X = np.asfortranarray(X)
    y = np.asfortranarray(y)

    last_K_coef = np.empty((K + 1, n_features))
    U = np.zeros((K, n_features))

    L = (X ** 2).sum(axis=0)

    for iter_idx in range(MAX_ITER):
        for j in range(n_features):
            tmp = coef
            tmp[j] = 0
            r = y - X @ tmp

            coef[j] = soft_thresh(coef[j] + X[:, j] @ r / L[j], alpha / L[j])

        gap, p_obj, d_obj = get_duality_gap(X, y, coef, alpha)
        gap_history_.append(gap)

        print(
            f"[{iter_idx+1}/{MAX_ITER}] p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} :: d_gap {gap:.5f}"
        )

        if use_acc:
            last_K_coef[iter_idx % (K + 1)] = coef

            if iter_idx % (K + 1) == K:
                U = np.zeros((K, n_features))

                for k in range(K):
                    U[k] = last_K_coef[k + 1] - last_K_coef[k]

                C = U @ U.T

                try:
                    z = np.linalg.solve(C, np.ones(K))
                    c = z / z.sum()

                    coef_acc = np.sum(
                        last_K_coef[:-1] * np.expand_dims(c, axis=-1), axis=0
                    )

                    p_obj_acc = primal(X, y, coef_acc, alpha)

                    if p_obj_acc < p_obj:
                        coef = coef_acc

                except np.linalg.LinAlgError:
                    print("LinAlg Error")

        if gap < TOL:
            print(f"Fitting ended after iteration {iter_idx + 1}.")
            break

    fig = plt.figure()

    plt.plot(np.log(gap_history_), label="Gap")

    plt.legend()

    plt.xlabel("Iteration")
    plt.ylabel("$p^* - d^*$ (logarithmic)")

    plt.title("Convergence speed", fontsize=13)

    fig.show()
