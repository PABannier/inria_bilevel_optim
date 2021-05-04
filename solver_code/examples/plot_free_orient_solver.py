import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from mtl.utils_datasets import compute_alpha_max
from mtl.simulated_data import simulate_data

from solver_lasso.solver_free_orient import MultiTaskLassoOrientation

X, Y, W, _ = simulate_data(
    n_samples=100, n_features=522, n_tasks=50, random_state=0
)

alpha_max = compute_alpha_max(X, Y)
alpha = alpha_max * 0.2

estimator_accelerated = MultiTaskLassoOrientation(
    alpha, n_orient=3, max_iter=7000, accelerated=True, verbose=True
)
estimator = MultiTaskLassoOrientation(
    alpha, n_orient=3, max_iter=7000, accelerated=False, verbose=True
)

# Fitting
estimator_accelerated.fit(X, Y)
estimator.fit(X, Y)

# Figure 1
fig = plt.figure()

# plt.plot(np.log(estimator.gap_history_), label="Gap - Non accelerated")
# plt.plot(np.log(estimator_accelerated.gap_history_), label="Gap - Accelerated")

plt.plot(estimator.primal_history_, label="Primal - Non accelerated")
plt.plot(estimator_accelerated.primal_history_, label="Primal - Accelerated")

plt.plot(estimator.dual_history_, label="Dual - Non accelerated")
plt.plot(estimator_accelerated.dual_history_, label="Dual - Accelerated")

plt.ylim((137.4, 137.7))

plt.legend()

plt.xlabel("Iteration")
plt.ylabel("$p^* - d^*$ (logarithmic)")

plt.title("Convergence speed", fontsize=13)

fig.show()
