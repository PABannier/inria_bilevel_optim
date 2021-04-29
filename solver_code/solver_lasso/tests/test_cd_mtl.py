import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state

from solver_lasso.utils import (
    block_soft_thresh,
    get_duality_gap_mtl,
    compute_alpha_max,
)

from celer import MultiTaskLasso


rng = check_random_state(0)

X = rng.randn(10, 15)
Y = rng.randn(10, 5)

n_samples, n_features = X.shape
_, n_tasks = Y.shape

alpha_max = compute_alpha_max(X, Y)
print("Alpha max:", alpha_max)

alpha = alpha_max * 0.1

coef = np.zeros((n_features, n_tasks))

L = (X ** 2).sum(axis=0)

for iter_idx in range(100):
    for j in range(n_features):
        tmp = coef
        tmp[j, :] = 0
        R = Y - X @ tmp

        coef[j, :] = block_soft_thresh(
            coef[j, :] + X[:, j] @ R / L[j], alpha / L[j]
        )

        # coef[j, :] = BST(
        #    coef[j, :] + x_j @ (Y - X @ coef) / L[j], alpha / L[j]
        # )

        gap, p_obj, d_obj = get_duality_gap_mtl(X, Y, coef, alpha)
        print(gap)


clf = MultiTaskLasso(alpha / (n_samples))
clf.fit(X, Y)

np.testing.assert_allclose(coef, clf.coef_.T, atol=1e-5)
