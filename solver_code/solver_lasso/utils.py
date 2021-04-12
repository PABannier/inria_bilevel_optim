import numpy as np
from numpy.linalg import norm
from numba import njit


@njit
def ST(x, lambda_):
    """Soft-thresholding operator"""
    return np.sign(x) * np.maximum(0.0, np.abs(x) - lambda_)


@njit
def BST(x, lambda_):
    """Block soft-thresholding operator"""
    a = 1 - lambda_ / norm(x)
    return np.maximum(0, a) * x


@njit
def norm_l21(X):
    res = 0
    for j in range(X.shape[0]):
        res += norm(X[j, :])
    return res


@njit
def norm_l2inf(X):
    res = 0
    for j in range(X.shape[0]):
        res = max(res, norm(X[j, :]))
    return res


# @njit
def prox_l21(X, lambda_):
    """Proximal operator for
    the l2,1 matrix norm"""
    x = norm(X, axis=1)
    return X * np.maximum(1 - lambda_ / np.expand_dims(x, axis=-1), 0)


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


def primal_mtl(R, coef, alpha):
    """Primal objective function for multi-task
    LASSO
    """
    p_obj = norm(R) ** 2
    p_obj += alpha * norm_l21(coef)
    return p_obj


def dual_mtl(X, Y, coef, alpha):
    """Dual objective function for multi-task
    LASSO
    """
    R = Y - X @ coef
    Theta = R / alpha
    d_norm_theta = np.max(norm(X.T @ Theta, axis=1))
    Theta /= d_norm_theta

    d_obj = norm(Y) ** 2 / 2
    d_obj -= ((alpha ** 2) / 2) * norm(Theta - Y / alpha) ** 2
    return d_obj

    # n_samples, n_tasks = Y.shape
    # d_obj = norm(Theta) ** 2 + np.trace(Theta.T @ Y)
    # return d_obj
    """
    R = y - X @ coef
    theta = R / alpha
    d_norm_theta = np.max(np.abs(X.T @ theta))
    theta /= d_norm_theta

    d_obj = (norm(y) ** 2) / 2
    d_obj -= ((alpha ** 2) / 2) * norm(theta - y / alpha) ** 2
    return d_obj
    """


def get_duality_gap_mtl(X, Y, coef, alpha):
    R = Y - X @ coef
    p_obj = primal_mtl(R, coef, alpha)
    d_obj = dual_mtl(X, Y, coef, alpha)
    return p_obj - d_obj, p_obj, d_obj


@njit
def fista_iteration(coef, X, y, t, z, L, alpha):
    coef_old = coef.copy()
    z = z + X.T @ (y - X @ z) / L
    coef = ST(z, alpha / L)

    t0 = t
    t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
    z = coef + ((t0 - 1) / t) * (coef - coef_old)
    return coef, t, z


@njit
def ista_iteration(coef, X, y, L, alpha):
    # coef = coef_.copy()
    coef += X.T @ (y - X @ coef) / L
    coef = ST(coef, alpha / L)
    return coef


@njit
def cd_iteration(n_features, X, coef, y, alpha, L):
    for j in range(n_features):
        tmp = coef
        tmp[j] = 0
        r = y - X @ tmp

        coef[j] = ST(coef[j] + X[:, j] @ r / L[j], alpha / L[j])
    return coef


@njit
def anderson_extrapolation(X, y, coef, U, last_K_coef, p_obj, alpha, K):
    """Anderson extrapolation

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Design matrix

    y : np.ndarray of shape (n_samples)
        Target vector

    coef : np.ndarray of shape (n_features)
        Coefficient vector

    U : np.ndarray of shape (K, n_features)

    last_K_coef : np.ndarray of shape (K+1, n_features)
        Stores the last K coefficient vectors

    p_obj : float
        Value of the primal problem without acceleration

    alpha : float
        Regularizing hyperparameter

    K : int
        Number of previous iterates used to extrapolate
    """
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
