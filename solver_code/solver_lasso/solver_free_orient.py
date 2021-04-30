import numpy as np
from numpy.linalg import norm

from numba import njit

from sklearn.utils import check_random_state
from mtl.utils_datasets import compute_alpha_max

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import check_random_state


def sum_squared(X):
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def groups_norm2(A, n_orient):
    """Compute squared L2 norms of groups inplace."""
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l21(A, n_orient, copy=True):
    """L21 norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sum(np.sqrt(groups_norm2(A, n_orient)))


def norm_l2inf(A, n_orient, copy=True):
    """L2-inf norm."""
    if A.size == 0:
        return 0.0
    if copy:
        A = A.copy()
    return np.sqrt(np.max(groups_norm2(A, n_orient)))


def primal_l21(M, G, X, active_set, alpha, n_orient):
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty
    return p_obj


def dgap_l21(M, G, X, active_set, alpha, n_orient):
    GX = np.dot(G[:, active_set], X)
    R = M - GX
    penalty = norm_l21(X, n_orient, copy=True)
    nR2 = sum_squared(R)
    p_obj = 0.5 * nR2 + alpha * penalty

    dual_norm = norm_l2inf(np.dot(G.T, R), n_orient, copy=False)
    scaling = alpha / dual_norm
    scaling = min(scaling, 1.0)
    d_obj = (scaling - 0.5 * (scaling ** 2)) * nR2 + scaling * np.sum(R * GX)

    gap = p_obj - d_obj
    return gap, p_obj, d_obj, R


def compute_lipschitz_constants(X, n_positions, n_orient):
    lc = np.empty(n_positions)
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        lc[j] = norm(X[:, idx], ord=2) ** 2
    return lc


class MultiTaskLasso(BaseEstimator, RegressorMixin):
    """Solver Multi-Task Lasso for neuroscience inverse
    problem. It supports fixed (n_orient=1) and free (n_orient > 1)
    orientiations.

    Parameters
    ----------
    n_orient: int, default=1
        Number of orientiation for a dipole. Fixed orientation
        corresponds to n_orient = 1. Free orientation corresponds
        to n_orient > 1.

    max_iter: int, default=100
        Maximum number of iterations for coordinate descent.

    tol: float, default=1e-5
        Gap threshold to stop the solver.

    warm_start: bool, default=True
        If True and if self.coef_ is not None, the solver
        starts its first iteration with the coefficients
        it has in memory.

    verbose: bool, default=False
        Verbosity.

    Attributes
    ----------
    coef_: array of shape (n_features, n_times)
        Contains the coefficients of the Lasso.

    gap_history_: list
        Stores the duality gap during fitting.
    """

    def __init__(
        self,
        n_orient=1,
        max_iter=1000,
        tol=1e-5,
        warm_start=True,
        verbose=False,
    ):
        self.n_orient = n_orient
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.verbose = verbose

        self.coef_ = None
        self.gap_history_ = []

    def fit(self, X, Y, alpha):
        X, Y = check_X_y(X, Y, multi_output=True)

        self.gap_history_ = []

        n_samples, n_features = X.shape
        n_samples, n_times = Y.shape
        n_positions = n_features // self.n_orient

        if self.warm_start and self.coef_:
            coef = self.coef_
            R = Y - np.dot(X, coef)
        else:
            coef = np.zeros((n_features, n_times))
            R = Y.copy()

        active_set = np.zeros(n_features, dtype=bool)

        lipschitz_constants = compute_lipschitz_constants(
            X, n_positions, self.n_orient
        )
        alpha_lc = alpha / lipschitz_constants
        one_over_lc = 1.0 / lipschitz_constants

        X = np.asfortranarray(X)

        # Checking arrays are Fortran-contiguous
        assert coef.T.flags.f_contiguous
        assert R.T.flags.f_contiguous
        assert X.flags.f_contiguous

        list_X_j_c = []
        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            list_X_j_c.append(np.ascontiguousarray(X[:, idx]))

        for i in range(self.max_iter):
            self._bcd(
                X,
                coef,
                R,
                active_set,
                one_over_lc,
                self.n_orient,
                n_positions,
                alpha_lc,
                list_X_j_c,
            )

            if (i + 1) % 5 == 0:
                gap, p_obj, d_obj, _ = dgap_l21(
                    Y, X, coef[active_set], active_set, alpha, self.n_orient
                )

                self.gap_history_.append(gap)

                if gap < self.tol:
                    if self.verbose:
                        print(
                            "Convergence reached ! (gap: %s < %s)"
                            % (gap, self.tol)
                        )
                    break

                if self.verbose > 0:
                    print(
                        "Iteration %d :: p_obj %f :: d_obj %f :: dgap %f :: n_active %d"
                        % (
                            i + 1,
                            p_obj,
                            d_obj,
                            gap,
                            np.sum(active_set) / self.n_orient,
                        )
                    )

        if gap > self.tol:
            print("Threshold not reached (gap: %s > %s" % (gap, self.tol))

        self.coef_ = coef

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_

    def _bcd(
        self,
        X,
        coef,
        R,
        active_set,
        one_over_lc,
        n_orient,
        n_positions,
        alpha_lc,
        list_X_j_c,
    ):
        for j, X_j_c in enumerate(list_X_j_c):
            idx = slice(j * n_orient, (j + 1) * n_orient)
            coef_j = coef[idx]

            coef_j_new = X_j_c.T @ R * one_over_lc[j]
            was_non_zero = coef_j[0, 0] != 0

            if was_non_zero:
                R += np.dot(X_j_c, coef_j)
                coef_j_new += coef_j

            block_norm = np.sqrt(sum_squared(coef_j_new))

            if block_norm <= alpha_lc[j]:
                coef_j.fill(0.0)
                active_set[idx] = False
            else:
                shrink = max(1.0 - alpha_lc[j] / block_norm, 0.0)
                coef_j_new *= shrink
                R -= np.dot(X_j_c, coef_j_new)
                coef_j[:] = coef_j_new
                active_set[idx] = True
