import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt


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
