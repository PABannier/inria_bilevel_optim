import numpy as np
from numpy.linalg import norm

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from mtl.utils_datasets import (
    get_duality_gap_mtl,
    primal_mtl,
    groups_norm2,
    sum_squared,
    get_dgemm,
)


class MultiTaskLassoOrientation(BaseEstimator, RegressorMixin):
    """MultiTask Lasso solver specificially designed for
    neuroscience inverse problem. It supports fixed and free
    dipole orientations.

    Main features: active set strategy, Anderson acceleration.

    Note: GEMM functions are called in each BCD pass for faster
    iteration time. It directly calls the appropriate BLAS function
    and avoids creating an extra temporary variable.

    Parameters
    ----------
    alpha: float
        Regularization parameter.

    n_orient: int, default=1
        Number of orientation for a dipole. Fixed orientation
        corresponds to n_orient = 1. Free orientation corresponds
        to n_orient > 1.

    max_iter: int, default=2000
        Maximum number of iterations for coordinate descent.

    tol: float, default=1e-8
        Gap threshold to stop the solver.

    warm_start: bool, default=True
        Starts fitting with previously fitted regression
        coefficients.

    accelerated: bool, default=True
        Use Anderson acceleration to speed up convergence.

    K: int, default=5
        Number of previous coefficient matrix used to compute
        the extrapolated coefficient matrix in Anderson
        acceleration.

    active_set_size: int, default=50
        Size of active set increase at each iteration.

    normalized_alpha: bool, default=True
        This solver expects an unscaled alpha (no normalization
        by the number of samples or tasks). If True, this solver
        expects a scaled alpha and will unscale it on its own. If
        False, this solver expects an unscaled alpha and won't
        change it.

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
        alpha,
        n_orient=3,
        max_iter=2000,
        tol=1e-4,
        warm_start=True,
        accelerated=True,
        K=5,
        active_set_size=100,
        normalized_alpha=True,
        verbose=False,
    ):
        self.alpha = alpha
        self.n_orient = n_orient
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.accelerated = accelerated
        self.K = K
        self.active_set_size = active_set_size
        self.normalized_alpha = normalized_alpha
        self.verbose = verbose

        self.gap_history_ = []
        self.primal_history_ = []
        self.coef_ = None
        self.active_set_ = None

    def fit(self, X, Y):
        """Fits the MultiTaskLasso estimator to the X and Y
        matrices.

        The solver initializes an active set and gradually
        increases it by a fixed amount (active_set_size).

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Design matrix.

        Y: array of shape (n_samples, n_times)
            Target matrix.
        """
        X, Y = check_X_y(X, Y, multi_output=True)

        # # If needed, unscale alpha by the number of samples
        # if self.normalized_alpha:
        #     self.alpha *= len(X)

        n_samples, n_features = X.shape
        n_times = Y.shape[1]
        n_positions = n_features // self.n_orient

        lipschitz_consts = np.empty(n_positions)

        for j in range(n_positions):
            idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
            lipschitz_consts[j] = norm(X[:, idx], ord=2) ** 2

        if self.active_set_ is None or self.warm_start == False:
            active_set = np.zeros(n_features, dtype=bool)
        else:
            # Useful for warm starting active set
            active_set = self.active_set_

        idx_large_corr = np.argsort(
            groups_norm2(np.dot(X.T, Y), self.n_orient)
        )
        new_active_idx = idx_large_corr[-self.active_set_size :]

        if self.n_orient > 1:
            new_active_idx = (
                self.n_orient * new_active_idx[:, None]
                + np.arange(self.n_orient)[None, :]
            ).ravel()

        active_set[new_active_idx] = True
        as_size = np.sum(active_set)

        highest_d_obj = -np.inf

        if self.warm_start and self.coef_ is not None:
            if self.coef_.shape != (n_features, n_times):
                raise ValueError(
                    f"Wrong dimension for initialized coefficients. "
                    + f"Got {self.coef_shape}. Expected {(n_features, n_times)}"
                )
            coef_init = self.coef_[active_set]  # ????
        else:
            coef_init = None

        for k in range(self.max_iter):
            lipschitz_consts_tmp = lipschitz_consts[
                active_set[:: self.n_orient]
            ]

            coef, as_ = self._block_coordinate_descent(
                X[:, active_set], Y, lipschitz_consts_tmp, coef_init
            )

            active_set[active_set] = as_.copy()
            idx_old_active_set = np.where(active_set)[0]

            _, p_obj, d_obj = get_duality_gap_mtl(
                X, Y, coef, active_set, self.alpha, self.n_orient
            )

            highest_d_obj = max(highest_d_obj, d_obj)
            gap = p_obj - highest_d_obj

            self.gap_history_.append(gap)
            self.primal_history_.append(p_obj)

            if self.verbose:
                print(
                    f"[{k+1}/{self.max_iter}] p_obj {p_obj:.5f} :: d_obj {d_obj:.5f} "
                    + f":: d_gap {gap:.5f} :: n_active_start {as_size // self.n_orient} "
                    + f":: n_active_end {np.sum(active_set) // self.n_orient}"
                )

            if gap < self.tol:
                if self.verbose:
                    print("Convergence reached!")
                break

            if k < (self.max_iter - 1):
                R = Y - X[:, active_set] @ coef
                idx_large_corr = np.argsort(
                    groups_norm2(np.dot(X.T, R), self.n_orient)
                )
                new_active_idx = idx_large_corr[-self.active_set_size :]

                if self.n_orient > 1:
                    new_active_idx = (
                        self.n_orient * new_active_idx[:, None]
                        + np.arange(self.n_orient)[None, :]
                    )
                    new_active_idx = new_active_idx.ravel()

                active_set[new_active_idx] = True
                idx_active_set = np.where(active_set)[0]
                as_size = np.sum(active_set)
                coef_init = np.zeros((as_size, n_times), dtype=coef.dtype)
                idx = np.searchsorted(idx_active_set, idx_old_active_set)
                coef_init[idx] = coef

        # Building full coefficient matrix and filling active set with
        # non-zero coefficients
        final_coef_ = np.zeros((len(active_set), n_times))
        if coef is not None:
            final_coef_[active_set] = coef

        self.coef_ = final_coef_
        self.active_set_ = active_set

        return self

    def predict(self, X):
        """If fitted, predicts X using the fitted coefficients.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        Y: array of shape (n_samples, n_times)
            Predicted matrix.
        """
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_

    def _block_coordinate_descent(self, X, Y, lipschitz, init):
        """Implements a block coordinate descent algorithm using
        Anderson acceleration.

        Parameters
        ----------
        X: array of shape (n_samples, n_features)
            Design matrix.

        Y: array of shape (n_samples, n_times)
            Target matrix.

        lipschitz: array (shape varies with active set)
            Lipschitz constants for every block.

        init: array (shape varies with active set)
            Coefficient initialized from previous iteration.
            If None, coefficients are initialized with zeros.

        Returns
        -------
        coef: array of shape (n_features, n_times)
            Coefficient matrix.

        active_set: array of shape (n_features)
            Contains boolean values. If True, the feature
            is active.
        """
        n_samples, n_times = Y.shape
        n_samples, n_features = X.shape
        n_positions = n_features // self.n_orient

        if init is None:
            coef = np.zeros((n_features, n_times))
            R = Y.copy()
        else:
            coef = init
            R = Y - X @ coef

        X = np.asfortranarray(X)

        if self.accelerated:
            last_K_coef = np.empty((self.K + 1, n_features, n_times))
            U = np.zeros((self.K, n_features * n_times))

        highest_d_obj = -np.inf
        active_set = np.zeros(n_features, dtype=bool)

        for iter_idx in range(self.max_iter):
            coef_j_new = np.zeros_like(coef[: self.n_orient, :], order="C")
            dgemm = get_dgemm()

            for j in range(n_positions):
                idx = slice(j * self.n_orient, (j + 1) * self.n_orient)
                coef_j = coef[idx]
                X_j = X[:, idx]

                # coef_j_new = X_j.T @ R / L[j]
                dgemm(
                    alpha=1 / lipschitz[j],
                    beta=0.0,
                    a=R.T,
                    b=X_j,
                    c=coef_j_new.T,
                    overwrite_c=True,
                )

                if coef_j[0, 0] != 0:
                    # R += X_j @ coef_j
                    dgemm(
                        alpha=1.0,
                        beta=1.0,
                        a=coef_j.T,
                        b=X_j.T,
                        c=R.T,
                        overwrite_c=True,
                    )
                    coef_j_new += coef_j

                block_norm = np.sqrt(sum_squared(coef_j_new))
                alpha_lc = self.alpha / lipschitz[j]

                if block_norm <= alpha_lc:
                    coef_j.fill(0.0)
                    active_set[idx] = False
                else:
                    shrink = max(1.0 - alpha_lc / block_norm, 0.0)
                    coef_j_new *= shrink

                    # R -= np.dot(X_j, coef_j_new)
                    dgemm(
                        alpha=-1.0,
                        beta=1.0,
                        a=coef_j_new.T,
                        b=X_j.T,
                        c=R.T,
                        overwrite_c=True,
                    )
                    coef_j[:] = coef_j_new
                    active_set[idx] = True

            _, p_obj, d_obj = get_duality_gap_mtl(
                X, Y, coef[active_set], active_set, self.alpha, self.n_orient
            )
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj

            # self.gap_history_.append(gap)

            if self.verbose:
                print(
                    f"[{iter_idx+1}/{self.max_iter}] p_obj {p_obj:.5f} :: "
                    + f"d_obj {d_obj:.5f} :: d_gap {gap:.5f}"
                )

            if self.accelerated:
                last_K_coef[iter_idx % (self.K + 1)] = coef

                if iter_idx % (self.K + 1) == self.K:
                    for k in range(self.K):
                        U[k] = (
                            last_K_coef[k + 1].ravel() - last_K_coef[k].ravel()
                        )

                    C = U @ U.T

                    try:
                        z = np.linalg.solve(C, np.ones(self.K))
                        c = z / z.sum()

                        coef_acc = np.sum(
                            last_K_coef[:-1] * c[:, None, None], axis=0
                        )

                        active_set_acc = norm(coef_acc, axis=1) != 0

                        p_obj_acc = primal_mtl(
                            X,
                            Y,
                            coef_acc[active_set_acc],
                            active_set_acc,
                            self.alpha,
                            self.n_orient,
                        )

                        if p_obj_acc < p_obj:
                            coef = coef_acc
                            active_set = active_set_acc
                            R = Y - X[:, active_set] @ coef[active_set]

                    except np.linalg.LinAlgError:
                        if self.verbose:
                            print("LinAlg Error")

            if gap < self.tol:
                if self.verbose:
                    print(f"Fitting ended after iteration {iter_idx + 1}.")
                break

        coef = coef[active_set]
        return coef, active_set
