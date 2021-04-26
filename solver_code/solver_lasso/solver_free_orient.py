import numpy as np
from numpy.linalg import norm

from sklearn.utils import check_random_state
from mtl.utils_datasets import compute_alpha_max

import ipdb


# Utils functions


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def group_norm_2(A, n_orient):
    """Computes a group norm (l2-norm over the columns)
    taking into account the dipole orientiation.
    """
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l_2_1(A, n_orient):
    """Mixed l_2_1 norm for matrix"""
    return np.sum(np.sqrt(group_norm_2(A, n_orient)))


def norm_l_2_inf(A, n_orient):
    """Mixed l_2_inf norm for matrix
    Note: l_2_inf is the dual norm of l_2_1
    """
    return np.sqrt(np.max(group_norm_2(A, n_orient)))


def block_soft_thresh(Y, alpha, n_orient):
    """Proximal operator for l_2_1 norm, also called
    block soft thresholding.
    """
    n_positions = Y.shape[0] // n_orient
    # rows_norm = np.sqrt(group_norm_2(Y, n_orient))
    rows_norm = np.sqrt(
        (Y * Y.conj()).real.reshape(n_positions, -1).sum(axis=1)
    )

    # rows_norm = np.sqrt(sum_squared(Y))

    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)

    if n_orient > 1:
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()

    if isinstance(shrink, float):
        Y *= shrink
    else:
        Y *= shrink[:, np.newaxis]

    return Y


def get_duality_gap(X, coef, Y, alpha, n_orient):
    Y_hat = X @ coef
    R = Y - Y_hat

    penalty = norm_l_2_1(coef, n_orient)

    # Why not Frobenius norm?
    # R_flat = R.ravel(order="F" if np.isfortran(R) else "C")
    # nR2 = np.dot(R_flat, R_flat)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    # Dual
    dual_norm = norm_l_2_inf(X.T @ R, n_orient)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)

    d_obj = (scaling - 0.5 * (scaling ** 2)) * norm(R) ** 2 + scaling * np.sum(
        R * Y_hat
    )

    gap = p_obj - d_obj
    return gap, p_obj, d_obj


def _bcd(X, coef, Y, list_X_j_c, lipschitz_consts, n_orient):
    """Implements one full pass of BCD.
    BCD stands for Block Coordinate Descent.
    """
    R = Y - X @ coef
    coef_j_new = np.zeros_like(X[0:n_orient, :], order="C")

    for j, X_j_c in enumerate(list_X_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        X_j = X[:, idx]
        coef_j = coef[idx, :]

        coef_j_new = X_j.T @ R  # / lipschitz_consts[j]
        coef_j_new += coef_j

        coef[idx, :] = block_soft_thresh(
            coef_j_new, alpha / lipschitz_consts[j], n_orient
        )

        # alpha_lc = alpha / lipschitz_consts[j]

        # coef[idx, :] = block_soft_thresh(coef_j_new, alpha_lc, n_orient)

    return coef


if __name__ == "__main__":
    rng = check_random_state(0)

    X = rng.randn(10, 15)
    Y = rng.randn(10, 5)

    X = np.asfortranarray(X)

    MAX_ITER = 100
    CHECK_DGAP_FREQ = 5
    TOL = 1e-8

    n_samples, n_features = X.shape
    _, n_tasks = Y.shape
    n_orient = 1

    n_positions = n_features // n_orient

    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max:", alpha_max)

    alpha = alpha_max * 0.1

    coef_ = np.zeros((n_features, n_tasks))

    lipschitz_consts = (X ** 2).sum(axis=0)

    # Storing list of contiguous arrays
    list_X_j_c = []
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        list_X_j_c.append(np.ascontiguousarray(X[:, idx]))

    for iter_idx in range(MAX_ITER):
        # ipdb.set_trace()

        coef_ = _bcd(X, coef_, Y, list_X_j_c, lipschitz_consts, n_orient)

        if iter_idx % CHECK_DGAP_FREQ == 0:
            gap, p_obj, d_obj = get_duality_gap(X, coef_, Y, alpha, n_orient)
            print(
                f"Iteration {iter_idx + 1} :: {p_obj:.5f} :: {gap:.5f} :: {d_obj:.5f}"
            )

            if gap < TOL:
                print("Threshold reached. Stopped fitting.")
                break
