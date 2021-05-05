import os
import time
from tqdm import tqdm
from joblib import parallel_backend
from joblib import Parallel, delayed

import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

from sklearn import linear_model
from sklearn.linear_model import Lasso, LassoLars
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin, RegressorMixin


N_JOBS = 20  # -1
INNER_MAX_NUM_THREADS = 1

MEMMAP_FOLDER = "."
OUTPUT_FILENAME_MEMMAP = os.path.join(MEMMAP_FOLDER, "output_memmap")


def _get_coef(est):
    """Get coefficients from a fitted regression estimator."""
    if hasattr(est, "steps"):
        return est.steps[-1][1].coef_
    return est.coef_


class SparseRegressor(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Provided regression estimator (ie model) solves inverse problem
    using data X and lead field L. The estimated coefficients (est_coef
    sometimes called z) are then used to predict which parcels are active.

    X must be of a specific structure with a column name 'subject' and
    'L_path' which gives the path to lead_field files for each subject
    """

    def __init__(self, model, n_jobs=1):
        self.model = model
        self.n_jobs = n_jobs
        self.parcel_indices = {}
        self.Ls = {}

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)

    def _run_model(self, model, L, X):
        norms = np.linalg.norm(L, axis=0)
        L = L / norms[None, :]

        """
        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx in tqdm(range(len(X)), total=len(X)):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(_get_coef(model))
            # est_coef /= norms
            est_coefs[idx] = est_coef
        """

        est_coefs = np.memmap(
            OUTPUT_FILENAME_MEMMAP,
            dtype=np.float32,
            shape=(X.shape[0], L.shape[1]),
            mode="w+",
        )

        start_time = time.time()

        with parallel_backend(
            "loky", inner_max_num_threads=INNER_MAX_NUM_THREADS
        ):
            Parallel(N_JOBS)(
                delayed(self._fit_model_for_sample)(
                    L, X, idx, model, est_coefs
                )
                for idx in range(len(X))
            )

        print("Fitting duration:", time.time() - start_time)

        return est_coefs.T

    def _fit_model_for_sample(self, L, X, idx, model, est_coefs):
        if idx % 10 == 0:
            print(f"Fitting idx #{idx}")
        x = X.iloc[idx].values
        model.fit(L, x)
        est_coef = np.abs(_get_coef(model))
        # est_coef /= norms
        est_coefs[idx] = est_coef

    def decision_function(self, X):
        X = X.reset_index(drop=True)

        for subject_id in np.unique(X["subject"]):
            if subject_id not in self.Ls:
                # load corresponding L if it's not already in
                L_used = X[X["subject"] == subject_id]["L_path"].iloc[0]
                lead_field = np.load(L_used)
                self.parcel_indices[subject_id] = lead_field["parcel_indices"]

                # scale L to avoid tiny numbers
                self.Ls[subject_id] = 1e8 * lead_field["lead_field"]
                assert (
                    self.parcel_indices[subject_id].shape[0]
                    == self.Ls[subject_id].shape[1]
                )

        n_parcels = np.max([np.max(s) for s in self.parcel_indices.values()])
        betas = np.empty((len(X), n_parcels))
        for subj_idx in np.unique(X["subject"]):
            L_used = self.Ls[subj_idx]

            X_used = X[X["subject"] == subj_idx]
            X_used = X_used.drop(["subject", "L_path"], axis=1)

            est_coef = self._run_model(self.model, L_used, X_used)

            beta = (
                pd.DataFrame(np.abs(est_coef))
                .groupby(self.parcel_indices[subj_idx])
                .max()
                .transpose()
            )
            betas[X["subject"] == subj_idx] = np.array(beta)
        return betas


# ====================================================================
# ============================= UTILS ================================
# ====================================================================


class CustomSparseEstimator(BaseEstimator, RegressorMixin):
    """Regression estimator which uses LassoLars algorithm with given alpha
    normalized for each lead field L and x."""

    def __init__(self, alpha=0.2, n_lasso_iter=10):
        self.alpha = alpha
        self.n_lasso_iter = n_lasso_iter

    def fit(self, L, x):
        L = StandardScaler().fit_transform(L)

        # Choosing alpha with SURE
        alpha_max = abs(L.T.dot(x)).max() / len(L)
        alpha_grid = np.geomspace(alpha_max * 0.5, alpha_max * 0.1, 20)
        criterion = SUREForAdaptiveLasso(1, alpha_grid)
        _, best_alpha = criterion.get_val(L, x)

        # Fitting with alpha
        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w: 1.0 / (
            2.0 * np.sqrt(np.abs(w)) + np.finfo(float).eps
        )

        n_samples, n_features = L.shape
        weights = np.ones(n_features)

        for k in range(self.n_lasso_iter):
            L_w = L / weights[np.newaxis, :]
            # alpha_max = abs(L.T.dot(x)).max() / len(L)
            # alpha = self.alpha * alpha_max
            clf = linear_model.LassoLars(
                alpha=best_alpha, fit_intercept=False, normalize=False
            )
            clf.fit(L_w, x)
            coef_ = clf.coef_ / weights
            weights = gprime(coef_)

        self.coef_ = coef_


def get_estimator():
    # Ls, parcel_indices = get_leadfields()
    custom_model = CustomSparseEstimator(alpha=0.2, n_lasso_iter=6)
    adaptive_lasso = SparseRegressor(custom_model)

    return adaptive_lasso


class SUREForAdaptiveLasso:
    def __init__(
        self,
        sigma,
        alpha_grid,
        n_iterations=5,
        penalty=None,
        random_state=None,
    ):
        self.sigma = sigma
        self.alpha_grid = alpha_grid
        self.n_iterations = n_iterations
        self.random_state = random_state

        self.n_alphas = len(self.alpha_grid)
        self.sure_path_ = np.empty((self.n_alphas))

        self.eps = None
        self.delta = None

        if penalty:
            self.penalty = penalty
        else:
            self.penalty = lambda u: 1.0 / (
                2 * np.sqrt(np.abs(u) + np.finfo(float).eps)
            )

    def get_val(self, X, y):
        n_samples = y.shape[0]

        if self.eps is None or self.delta is None:
            self._init_eps_and_delta(n_samples)

        X, y = check_X_y(X, y)

        coefs_grid_1, coefs_grid_2 = self._fit_reweighted_with_grid(X, y)

        for i, (coef1, coef2) in enumerate(zip(coefs_grid_1, coefs_grid_2)):
            sure_val = self._compute_sure_val(coef1, coef2, X, y)
            self.sure_path_[i] = sure_val

        best_sure_ = np.min(self.sure_path_)
        best_alpha_ = self.alpha_grid[np.argmin(self.sure_path_)]

        # plot_sure_path(self.alpha_grid, self.sure_path_)

        return best_sure_, best_alpha_

    def _reweight_op(self, regressor, X, y, w):
        X_w = X / w[np.newaxis, :]
        regressor.fit(X_w, y)

        coef = (regressor.coef_ / w).T
        w = self.penalty(coef)

        return coef, w

    def _compute_sure_val(self, coef1, coef2, X, y):
        n_samples, n_features = X.shape

        dof = ((X @ (coef2 - coef1)) * self.delta).sum() / self.eps

        sure = norm(y - X @ coef1) ** 2
        sure -= n_samples * self.sigma ** 2
        sure += 2 * dof * self.sigma ** 2

        return sure

    def _fit_reweighted_with_grid(self, X, y):
        n_features = X.shape[1]

        coef1_0 = np.empty((self.n_alphas, n_features))
        coef2_0 = np.empty((self.n_alphas, n_features))

        y_eps = y + self.eps * self.delta

        regressor1 = Lasso(np.nan, fit_intercept=False, warm_start=True)
        regressor2 = Lasso(np.nan, fit_intercept=False, warm_start=True)

        for j, alpha in enumerate(self.alpha_grid):
            regressor1.alpha = alpha
            regressor2.alpha = alpha
            coef1_0[j] = regressor1.fit(X, y).coef_
            coef2_0[j] = regressor2.fit(X, y).coef_

        del regressor1
        del regressor2

        regressor1 = LassoLars(np.nan, fit_intercept=False)
        regressor2 = LassoLars(np.nan, fit_intercept=False)

        coefs_1_ = coef1_0.copy()
        coefs_2_ = coef2_0.copy()

        for j, alpha in enumerate(self.alpha_grid):
            regressor1.alpha = alpha
            regressor2.alpha = alpha

            w1 = self.penalty(coef1_0[j])
            w2 = self.penalty(coef2_0[j])

            for _ in range(self.n_iterations - 1):
                mask1 = w1 != 1.0 / np.finfo(float).eps
                mask2 = w2 != 1.0 / np.finfo(float).eps
                coefs_1_[j][~mask1] = 0.0
                coefs_2_[j][~mask2] = 0.0

                if mask1.sum():
                    coefs_1_[j][mask1], w1[mask1] = self._reweight_op(
                        regressor1, X[:, mask1], y, w1[mask1]
                    )
                if mask2.sum():
                    coefs_2_[j][mask2], w2[mask2] = self._reweight_op(
                        regressor2, X[:, mask2], y_eps, w2[mask2]
                    )
        return coefs_1_, coefs_2_

    def _init_eps_and_delta(self, n_samples):
        rng = check_random_state(self.random_state)
        self.eps = 2 * self.sigma / (n_samples ** 0.3)
        self.delta = rng.randn(n_samples)
