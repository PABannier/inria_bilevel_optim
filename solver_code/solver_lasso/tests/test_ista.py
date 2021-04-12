import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso
from sklearn.utils import check_random_state

from solver_lasso.utils import ST, primal, dual, compute_alpha_max
from solver_lasso.ista import ProxLasso


rng = check_random_state(0)

n_samples = 20
n_features = 50
X = rng.randn(n_samples, n_features)
y = rng.randn(n_samples)

alpha_max = compute_alpha_max(X, y)
print(alpha_max)
alpha = alpha_max / 10
max_iter = 10000
tol = 1e-6


def test_ista_support_recovery():
    clf1 = ProxLasso(alpha, max_iter, tol=tol, accelerated=False)
    clf1.fit(X, y)

    clf = Lasso(alpha=alpha / n_samples, fit_intercept=False, tol=tol)
    clf.fit(X, y)

    np.testing.assert_allclose(clf.coef_, clf1.coef_, atol=1e-4)


def test_fista_support_recovery():
    clf1 = ProxLasso(alpha, max_iter, tol=tol, accelerated=True)
    clf1.fit(X, y)

    clf = Lasso(alpha=alpha / n_samples, fit_intercept=False, tol=tol)
    clf.fit(X, y)

    np.testing.assert_allclose(clf.coef_, clf1.coef_, atol=1e-4)
