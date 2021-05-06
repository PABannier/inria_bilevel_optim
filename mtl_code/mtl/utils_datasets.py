import functools

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt


@functools.lru_cache(None)
def get_dgemm():
    from scipy import linalg

    return linalg.get_blas_funcs("gemm", (np.empty(0, np.float64),))


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


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def groups_norm2(A, n_orient=1):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def primal_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    """Primal objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


def dual_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    """Dual objective function for multi-task
    LASSO
    """
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    dual_norm = norm_l2_inf(np.dot(X.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(
        R * Y_hat
    )
    return d_obj


def get_duality_gap_mtl(X, Y, coef, active_set, alpha, n_orient=1):
    Y_hat = np.dot(X[:, active_set], coef)
    R = Y - Y_hat
    penalty = norm_l2_1(coef, n_orient, copy=True)
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


def compute_alpha_max(X, Y):
    if len(Y.shape) > 1:
        B = X.T @ Y
        b = norm(B, axis=1)
        return np.max(b) / X.shape[0]
    else:
        return np.max(np.abs(X.T @ Y)) / len(X)


def plot_sure_mse_path(alphas, alpha_max, mse_metrics, sure_metrics, mse_path):
    plt.figure()

    plt.semilogx(
        alphas / alpha_max,
        mse_metrics,
        linestyle="--",
        label="MSE",
        color="midnightblue",
    )
    plt.semilogx(
        alphas / alpha_max,
        sure_metrics,
        linestyle="--",
        label="SURE",
        color="orange",
    )

    min_idx = sure_metrics.argmin()
    plt.axvline(
        x=alphas[min_idx] / alpha_max,
        color="orange",
        linestyle="dashed",
        linewidth=3,
        label="Best SURE $\lambda$",
    )

    min_idx_2 = mse_path.mean(axis=1).argmin()
    plt.axvline(
        x=alphas[min_idx] / alpha_max,
        color="midnightblue",
        linestyle="dashed",
        linewidth=3,
        label="Best MSE $\lambda$",
    )

    plt.xlabel("$\lambda / \lambda_{\max}$", fontsize=12)
    plt.ylabel("MSE / SURE (normalized)", fontsize=12)
    plt.title("MSE vs SURE paths", fontsize=15, fontweight="bold")
    plt.legend()
    plt.show(block=True)


def plot_original_reconstructed_signal(original, reconstructed, title):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4))

    axes[0].set_title("Original signal", fontweight="bold", fontsize=15)
    axes[0].imshow(original.T, cmap="binary")

    axes[1].set_title("Recovered signal", fontweight="bold", fontsize=15)
    axes[1].imshow(reconstructed.T, cmap="binary")

    fig.suptitle(
        title,
        fontweight="bold",
        fontsize=18,
    )

    plt.show(block=True)


def plot_original_reconstructed_signal_band(
    original, reconstructed, title="Sparsity patterns"
):
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), constrained_layout=True)

    axes[0].spy(original.T, aspect="auto")
    axes[0].xaxis.tick_bottom()
    axes[0].set_title("Original signal", fontsize=14)
    axes[0].set_ylabel("Tasks", fontsize=12)

    axes[1].spy(reconstructed.T, aspect="auto")
    axes[1].xaxis.tick_bottom()
    axes[1].set_title("Reconstructed signal", fontsize=14)

    plt.suptitle(title, fontweight="bold", fontsize=20)
    plt.ylabel("Tasks", fontsize=12)
    plt.xlabel("Features", fontsize=12)
    plt.show(block=True)
