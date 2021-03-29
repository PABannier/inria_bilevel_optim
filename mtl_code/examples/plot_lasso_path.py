import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.simulated_data import simulate_data


X, Y, _ = simulate_data(
    n_samples=50, n_features=250, n_tasks=25, nnz=10, corr=0.2, random_state=2020, snr=3
)

alpha_max = np.max(norm(X.T @ Y, axis=1)) / X.shape[0]
print("Alpha max for large experiment:", alpha_max)

n_folds = 5
alphas = np.geomspace(alpha_max, alpha_max / 100, num=30)
regressor_cv = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
regressor_cv.fit(X, Y)


plt.figure(figsize=(8, 6))
for idx_fold in range(n_folds):
    plt.semilogx(alphas, regressor_cv.mse_path_[:, idx_fold], linestyle="--")
plt.semilogx(alphas, regressor_cv.mse_path_.mean(axis=1), linewidth=4, color="black")
plt.xlabel("alpha", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.show(block=True)
