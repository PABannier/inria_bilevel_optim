import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state

from solver_lasso.utils import get_duality_gap_mtl, compute_alpha_max, prox_l21

from celer import MultiTaskLasso

rng = check_random_state(0)

X = rng.randn(5, 10)
Y = rng.randn(5, 5)

n_samples, n_features = X.shape
_, n_tasks = Y.shape

L = norm(X, ord=2) ** 2

alpha_max = compute_alpha_max(X, Y)
print("Alpha max:", alpha_max)

alpha = alpha_max * 0.1

coef = np.zeros((n_features, n_tasks))

for iter_idx in range(1000):
    coef += X.T @ (Y - X @ coef) / L
    coef = prox_l21(coef, alpha / L)
    gap, p_obj, d_obj = get_duality_gap_mtl(X, Y, coef, alpha)
    if gap < 1e-4:
        print("Threshold hit. Fitting done.")
        break
    print(
        f"[{iter_idx}/1000] Primal: {p_obj:.4f}, Dual: {d_obj:.4f}, Gap: {gap:.4f}"
    )

clf = MultiTaskLasso(alpha / n_samples, max_iter=1000)
clf.fit(X, Y)

np.testing.assert_allclose(coef, clf.coef_.T, atol=1e-4)