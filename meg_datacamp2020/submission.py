import os
from tqdm.notebook import tqdm

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin

from sklearn.metrics import jaccard_score

from mtl.sure import SURE
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.utils_datasets import compute_alpha_max


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

        est_coefs = np.empty((X.shape[0], L.shape[1]))
        for idx, idx_used in enumerate(X.index.values):
            x = X.iloc[idx].values
            model.fit(L, x)
            est_coef = np.abs(_get_coef(model))
            est_coef /= norms  # ??????
            est_coefs[idx] = est_coef

        return est_coefs.T

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


class CustomSparseEstimator(BaseEstimator, RegressorMixin):
    """Regression estimator which uses Reweighted Multi-Task Lasso
    estimator and automatically selects the optimal alpha based on
    SURE minimization.
    """

    def __init__(self):
        self.best_sure_ = np.inf
        self.best_alpha_ = 0
        self.estimator_ = None

    def fit(self, L, x):
        alpha_max = compute_alpha_max(L, x)
        alphas = np.geomspace(alpha_max, alpha_max / 50, 50)

        # Sigma = 1 confirmé???????

        for alpha in tqdm(alphas, total=len(alphas)):
            estimator = SURE(ReweightedMultiTaskLasso, 1, random_state=0)
            sure_val_ = estimator.get_val(L, x, alpha)
            if sure_val_ < self.best_sure_:
                self.best_sure_ = sure_val_
                self.best_alpha_ = alpha

        print("best sure", self.best_sure_)
        print("best alpha", self.best_alpha_)

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha_)
        estimator.fit(L, x)

        self.coef_ = estimator.coef_


def get_estimator():
    custom_model = CustomSparseEstimator()
    adaptive_lasso = SparseRegressor(custom_model)
    return adaptive_lasso
