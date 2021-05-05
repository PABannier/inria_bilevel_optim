import numpy as np
from numpy.linalg import norm
from numba import njit


@njit
def soft_thresh(x, alpha):
    """Soft-thresholding operator"""
    return np.sign(x) * np.maximum(0.0, np.abs(x) - alpha)


@njit
def block_soft_thresh(x, alpha):
    """Block soft-thresholding operator"""
    norm_x = norm(x)
    if norm_x < alpha:
        return np.zeros_like(x)
    else:
        return (1 - alpha / norm_x) * x


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def groups_norm2(A, n_orient=1):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l2_1(X, n_orient, copy=True):
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sum(np.sqrt(groups_norm2(X, n_orient)))


def norm_l2_inf(X, n_orient, copy=True):
    if X.size == 0:
        return 0.0
    if copy:
        X = X.copy()
    return np.sqrt(np.max(groups_norm2(X, n_orient)))


@njit
def primal(X, y, coef, alpha):
    """Primal objective function for single-task
    LASSO
    """
    p_obj = (norm(y - X @ coef) ** 2) / 2
    p_obj += alpha * norm(coef, ord=1)
    return p_obj


@njit
def dual(X, y, coef, alpha):
    """Dual objective function for single-task
    LASSO
    """
    R = y - X @ coef
    theta = R / alpha
    d_norm_theta = np.max(np.abs(X.T @ theta))
    theta /= d_norm_theta

    d_obj = (norm(y) ** 2) / 2
    d_obj -= ((alpha ** 2) / 2) * norm(theta - y / alpha) ** 2
    return d_obj


@njit
def get_duality_gap(X, y, coef, alpha):
    """Computes the duality gap for single-task
    LASSO
    """
    p_obj = primal(X, y, coef, alpha)
    d_obj = dual(X, y, coef, alpha)
    return p_obj - d_obj, p_obj, d_obj


def primal_mtl(X, Y, coef, alpha, n_orient=1):
    """Primal objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X, coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


def dual_mtl(X, coef, Y, alpha, n_orient=1):
    """Dual objective function for multi-task
    LASSO
    """
    dual_norm = norm_l2_inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    return d_obj


def get_duality_gap_mtl(X, Y, coef, alpha, n_orient=1):
    Y_hat = np.dot(X, coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    dual_norm = norm_l2_inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )

    gap = p_obj - d_obj
    return gap, p_obj, d_obj


def fista_iteration(coef, X, y, t, z, L, alpha):
    coef_old = coef.copy()
    z = z + X.T @ (y - X @ z) / L
    coef = soft_thresh(z, alpha / L)

    t0 = t
    t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
    z = coef + ((t0 - 1) / t) * (coef - coef_old)
    return coef, t, z


def ista_iteration(coef, X, y, L, alpha):
    # coef = coef_.copy()
    coef += X.T @ (y - X @ coef) / L
    coef = soft_thresh(coef, alpha / L)
    return coef


@njit
def cd_iteration(n_features, X, coef, y, alpha, L):
    for j in range(n_features):
        tmp = coef
        tmp[j] = 0
        r = y - X @ tmp

        coef[j] = soft_thresh(coef[j] + X[:, j] @ r / L[j], alpha / L[j])
    return coef


# TODO maybe we can remove coeff from this function
@njit
def anderson_extrapolation(X, y, coef, last_K_coef, p_obj, alpha, K):
    """Anderson extrapolation

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix

    y : np.ndarray of shape (n_samples)
        Target vector

    coef : np.ndarray of shape (n_features)
        Regression coefficients

    last_K_coef : np.ndarray of shape (K+1, n_features)
        Stores the last K coefficient vectors

    p_obj : float
        Value of the primal problem without acceleration

    alpha : float
        Regularizing hyperparameter

    K : int
        Number of previous iterates used to extrapolate

    Returns
    -------
    coef : np.ndarray (n_features)
        Regression coefficients.
    """
    n_features = X.shape[1]
    U = np.zeros((K, n_features))
    for k in range(K):
        U[k] = last_K_coef[k + 1] - last_K_coef[k]

    C = U @ U.T

    try:
        z = np.linalg.solve(C, np.ones(K))
        c = z / z.sum()

        # coef_acc = np.sum(last_K_coef[:-1] * c[:, None], axis=0)
        coef_acc = np.sum(
            last_K_coef[:-1] * np.expand_dims(c, axis=-1), axis=0
        )

        p_obj_acc = primal(X, y, coef_acc, alpha)

        if p_obj_acc < p_obj:
            coef = coef_acc

    except:  # Numba does not support custom Numpy LinAlg exception
        print("LinAlg Error")

    return coef


def compute_alpha_max(X, y):
    """Computes maximum alpha"""
    if len(y.shape) == 1:
        return np.max(X.T @ y)
    elif len(y.shape) == 2:
        B = X.T @ y
        b = norm(B, axis=1)
        return np.max(b)
    else:
        raise ValueError(
            f"Incorect number of dimension for target. Must be 1 or 2, but got {len(y.shape)}"
        )
