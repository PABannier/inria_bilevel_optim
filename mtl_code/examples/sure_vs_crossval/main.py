import joblib

import numpy as np
from numpy.linalg import norm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, jaccard_score, mean_squared_error

from celer import MultiTaskLasso
from celer import MultiTaskLassoCV

from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.sure import SURE

from examples.utils import compute_alpha_max


def reconstruct_signal(corr, random_state=0):
    print(f"=== Simulating data for {corr} ===")
    X, Y, coef, sigma = simulate_data(
        n_samples=30, n_features=100, n_tasks=50, nnz=5, snr=1.5, corr=corr
    )

    n_folds = 5
    alpha_max = compute_alpha_max(X, Y)
    alphas = np.geomspace(alpha_max, alpha_max / 50, num=50)

    # Reweighted
    regressor = ReweightedMultiTaskLassoCV(alphas, n_folds=n_folds)
    regressor.fit(X, Y, coef)

    sure_path = []

    for alpha in alphas:
        estimator = SURE(
            ReweightedMultiTaskLasso, sigma, random_state=random_state
        )
        metric = estimator.get_val(X, Y, alpha, n_iterations=5)
        sure_path.append(metric)

    reweighted_scores = {
        "mse": regressor.mse_path_,
        "f1": regressor.f1_path_,
        "jaccard": regressor.jaccard_path_,
        "sure": sure_path,
        "alpha": alphas,
    }

    joblib.dump(reweighted_scores, f"data/scores_reweighted_corr_{corr}.pkl")

    # Non-reweighted
    regressor = MultiTaskLassoCV(alphas=alphas, cv=5)
    regressor.fit(X, Y)

    sure_path = []

    for alpha in alphas:
        estimator = SURE(MultiTaskLasso, sigma, random_state=random_state)
        metric = estimator.get_val(X, Y, alpha, n_iterations=5)
        sure_path.append(metric)

    lasso_scores = {"mse": regressor.mse_path_, "sure": sure_path}

    joblib.dump(lasso_scores, f"data/scores_lasso_corr_{corr}.pkl")


if __name__ == "__main__":
    for c in [0, 0.3, 0.5, 0.7, 0.9]:
        reconstruct_signal(c)