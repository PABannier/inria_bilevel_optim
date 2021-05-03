import numpy as np
import matplotlib.pyplot as plt

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.sure_warm_start import SUREForReweightedMultiTaskLasso
from mtl.utils_datasets import compute_alpha_max

if __name__ == "__main__":
    RANDOM_STATE = 0

    rng = np.random.RandomState(RANDOM_STATE)

    N_TASKS = 5
    N_FEATURES = 15
    N_SAMPLES = 10

    SIGMA = 2

    Y = SIGMA * rng.randn(N_SAMPLES, N_TASKS)
    X = rng.randn(N_SAMPLES, N_FEATURES)

    alpha_max = compute_alpha_max(X, Y)
    alpha_grid = np.geomspace(alpha_max, alpha_max / 10, 15)

    criterion = SUREForReweightedMultiTaskLasso(SIGMA, alpha_grid)
    best_sure, best_val = criterion.get_val(X, Y)

    plt.figure()

    plt.plot(alpha_grid / alpha_grid[0], criterion.sure_path_, label="SURE")
    plt.plot(alpha_grid / alpha_grid[0], criterion.dof_history_, label="DOF")
    plt.plot(
        alpha_grid / alpha_grid[0],
        criterion.data_fitting_history_,
        label="Data fitting",
    )
    plt.title("Alpha grid")
    plt.xlabel("$\lambda / \lambda_{max}$")
    plt.ylabel("SURE")
    plt.legend()
    plt.show()
