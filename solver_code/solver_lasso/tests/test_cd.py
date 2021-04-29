import numpy as np
from numpy.linalg import norm

from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.utils import check_random_state

from solver_lasso.utils import primal, dual, compute_alpha_max
from solver_lasso.cd import CDLasso


def test_support_recovery():
    rng = check_random_state(0)

    X = rng.randn(30, 100)
    y = rng.randn(30)

    n_samples, n_features = X.shape

    alpha_max = compute_alpha_max(X, y)
    print("Alpha max:", alpha_max)

    alpha = alpha_max * 0.3

    clf1 = Lasso(alpha=alpha / n_samples, fit_intercept=False)
    clf2 = CDLasso(alpha=alpha, accelerated=False)

    clf1.fit(X, y)
    clf2.fit(X, y)

    # Tester le support
    np.testing.assert_array_equal((clf1.coef_ != 0), (clf2.coef_ != 0))


def test_coefficient():
    rng = check_random_state(42)

    X = rng.randn(15, 30)
    y = rng.randn(15)

    n_samples, n_features = X.shape

    alpha_max = compute_alpha_max(X, y)
    print("Alpha max:", alpha_max)

    alpha = alpha_max * 0.3

    clf1 = Lasso(alpha=alpha / n_samples, fit_intercept=False)
    clf2 = CDLasso(alpha=alpha, accelerated=False)

    clf1.fit(X, y)
    clf2.fit(X, y)

    # Tester le support
    np.testing.assert_allclose(clf1.coef_, clf2.coef_, atol=1e-3)


def test_extrapolation():
    rng = check_random_state(0)

    X = rng.randn(30, 100)
    y = rng.randn(30)

    n_samples, n_features = X.shape

    alpha_max = compute_alpha_max(X, y)
    print("Alpha max:", alpha_max)

    alpha = alpha_max * 0.3

    clf1 = CDLasso(alpha)
    clf1.fit(X, y)

    clf2 = Lasso(alpha / n_samples, fit_intercept=False)
    clf2.fit(X, y)

    np.testing.assert_array_equal((clf1.coef_ != 0), (clf2.coef_ != 0))
    np.testing.assert_allclose(clf1.coef_, clf2.coef_, atol=1e-3)
