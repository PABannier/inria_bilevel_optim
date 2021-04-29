import numpy as np
from numpy.linalg import norm

from numba import njit

from sklearn.utils import check_random_state
from mtl.utils_datasets import compute_alpha_max

from celer import MultiTaskLasso


# Utils functions


def sum_squared(X):
    """Computes the Frobenius norm for a matrix
    in a more efficient way.
    """
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


def get_duality_gap(X, coef, Y, alpha, n_orient):
    Y_hat = X @ coef
    R = Y - Y_hat

    # Primal
    penalty = norm_l_2_1(coef, n_orient)
    nR2 = sum_squared(R) / np.prod(X.shape)
    p_obj = 0.5 * nR2 + alpha * penalty

    # Dual
    # Scalar chosen to make theta dual feasible
    dual_norm = norm_l_2_inf(X.T @ R, n_orient)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)

    d_obj = (
        scaling - np.prod(X.shape) * 0.5 * (scaling ** 2)
    ) * nR2 + scaling * np.sum(R * Y_hat)

    gap = p_obj - d_obj
    return gap, p_obj, d_obj


def block_soft_thresh(Y, alpha, n_orient):
    """Proximal operator for l_2_1 norm, also called
    block soft thresholding.
    """

    n_positions = Y.shape[0] // n_orient
    rows_norm = np.sqrt(
        (Y * Y.conj()).real.reshape(n_positions, -1).sum(axis=1)
    )

    shrink = np.maximum(1.0 - alpha / np.maximum(rows_norm, alpha), 0.0)

    if n_orient > 1:
        shrink = np.tile(shrink[:, None], [1, n_orient]).ravel()

    if isinstance(shrink, float):
        Y *= shrink
    else:
        Y *= shrink[:, np.newaxis]

    return Y


def _bcd(X, coef, Y, list_X_j_c, R, lipschitz_consts, n_orient, alpha):
    """Implements one full pass of BCD.
    BCD stands for Block Coordinate Descent.
    """
    for j, X_j_c in enumerate(list_X_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        coef_j = coef[idx].copy()

        coef_j_new = X_j_c.T @ R / -lipschitz_consts[j]
        coef_j_new += coef_j

        coef[idx, :] = block_soft_thresh(
            coef_j_new, alpha / lipschitz_consts[j], n_orient
        )

        R += X_j_c @ (coef[idx, :] - coef_j)


if __name__ == "__main__":
    rng = check_random_state(42)

    X = rng.randn(10, 15)
    Y = rng.randn(10, 5)

    MAX_ITER = 20
    CHECK_DGAP_FREQ = 1
    TOL = 1e-8

    n_samples, n_features = X.shape
    _, n_tasks = Y.shape
    n_orient = 3

    n_positions = n_features // n_orient

    alpha_max = compute_alpha_max(X, Y)
    print("Alpha max:", alpha_max)
    alpha = alpha_max * 0.1

    coef_ = np.zeros((n_features, n_tasks))

    # Storing list of contiguous arrays
    list_X_j_c = []
    lipschitz_consts = np.empty((n_positions))

    R = Y.copy()

    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        list_X_j_c.append(np.ascontiguousarray(X[:, idx]))
        lipschitz_consts[j] = norm(X[:, idx].T @ X[:, idx])

    for iter_idx in range(MAX_ITER):
        # ipdb.set_trace()
        _bcd(X, coef_, Y, list_X_j_c, R, lipschitz_consts, n_orient, alpha)

        if iter_idx % CHECK_DGAP_FREQ == 0:
            gap, p_obj, d_obj = get_duality_gap(X, coef_, Y, alpha, n_orient)
            print(
                f"Iteration {iter_idx + 1} :: {p_obj:.5f} :: {gap:.5f} :: {d_obj:.5f}"
            )

            if gap < TOL:
                print("Threshold reached. Stopped fitting.")
                break

    estimator = MultiTaskLasso(alpha, verbose=1, fit_intercept=False)
    estimator.fit(X, Y)
