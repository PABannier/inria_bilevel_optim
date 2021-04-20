import time
from tqdm import tqdm
import joblib

import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import matplotlib.pylab as pl

from sklearn.utils import check_random_state

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates

from mtl.simulated_data import simulate_data
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.mtl import ReweightedMultiTaskLasso
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max


def benchmark_on_simulated_data(n_samples=100, n_features=150, n_tasks=50):
    random_state = np.random.RandomState(0)

    X, Y, _, sigma = simulate_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=n_tasks,
        nnz=5,
        snr=2,
        corr=0.2,
        random_state=random_state,
    )

    alpha_max = compute_alpha_max(X, Y)
    alphas = np.geomspace(alpha_max, alpha_max / 10, 50)

    criterion = ReweightedMultiTaskLassoCV(
        alphas,
        n_iterations=5,
        random_state=random_state,
    )

    criterion2 = SURE(
        ReweightedMultiTaskLasso, sigma, random_state=random_state
    )

    start_cv = time.time()
    criterion.fit(X, Y)
    duration_cv = time.time() - start_cv

    start_sure = time.time()
    for alpha in tqdm(alphas, total=len(alphas)):
        sure_crit_ = criterion2.get_val(X, Y, alpha)
    duration_sure = time.time() - start_sure

    return duration_cv, duration_sure


def benchmark_on_dataset(condition):
    def load_data():
        data_path = sample.data_path()
        fwd_fname = (
            data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
        )
        ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
        cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
        subjects_dir = data_path + "/subjects"

        noise_cov = mne.read_cov(cov_fname)
        evoked = mne.read_evokeds(
            ave_fname, condition=condition, baseline=(None, 0)
        )
        evoked.crop(tmin=0.05, tmax=0.15)
        evoked = evoked.pick_types(eeg=False, meg=True)
        forward = mne.read_forward_solution(fwd_fname)

        return evoked, forward, noise_cov

    def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
        from mne.inverse_sparse.mxne_inverse import (
            _prepare_gain,
            is_fixed_orient,
            _reapply_source_weighting,
            _make_sparse_stc,
        )

        all_ch_names = evoked.ch_names

        # Handle depth weighting and whitening (here is no weights)
        (
            forward,
            gain,
            gain_info,
            whitener,
            source_weighting,
            mask,
        ) = _prepare_gain(
            forward,
            evoked.info,
            noise_cov,
            pca=False,
            depth=depth,
            loose=loose,
            weights=None,
            weights_min=None,
            rank=None,
        )

        sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
        M = evoked.data[sel]

        M = np.dot(whitener, M)

        n_orient = 1 if is_fixed_orient(forward) else 3
        X, active_set, duration_sure, duration_cv = solver(M, gain, n_orient)
        X = _reapply_source_weighting(X, source_weighting, active_set)

        stc = _make_sparse_stc(
            X,
            active_set,
            forward,
            tmin=evoked.times[0],
            tstep=1.0 / evoked.info["sfreq"],
        )

        return stc, duration_sure, duration_cv

    def solver(M, G, n_orient=1):
        alpha_max = compute_alpha_max(G, M)
        print("Alpha max:", alpha_max)

        rs = np.random.RandomState(42)

        alphas = np.geomspace(alpha_max, alpha_max / 10, num=50)
        n_folds = 5

        best_alpha_ = None
        best_sure_ = np.inf

        criterion = SURE(ReweightedMultiTaskLasso, 1, random_state=rs)

        start_sure = time.time()
        for alpha in tqdm(alphas, total=len(alphas)):
            sure_crit_ = criterion.get_val(G, M, alpha)
            if sure_crit_ < best_sure_:
                best_alpha_ = alpha
                best_sure_ = sure_crit_

        duration_sure = time.time() - start_sure
        print("SURE", duration_sure)

        criterion2 = ReweightedMultiTaskLassoCV(
            alpha_grid=alphas, n_folds=n_folds, warm_start=True
        )
        start_cv = time.time()
        criterion2.fit(G, M)
        duration_cv = time.time() - start_cv
        print("MSE", duration_cv)
        best_alpha_ = criterion2.best_alpha_

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha_, warm_start=True)
        estimator.fit(G, M)

        X = estimator.coef_

        active_set = norm(X, axis=1) != 0
        return X[active_set, :], active_set, duration_sure, duration_cv

    loose, depth = 0, 0.9
    evoked, forward, noise_cov = load_data()

    _, duration_sure, duration_cv = apply_solver(
        solver, evoked, forward, noise_cov, loose, depth
    )

    return duration_sure, duration_cv


if __name__ == "__main__":
    # Benchmarking on synthetic data
    # benchmarks_n_samples = {}
    # n_samples_to_test = [25, 50, 75, 100, 150, 200]

    # for n_samples in tqdm(n_samples_to_test, total=len(n_samples_to_test)):
    #     duration_cv, duration_sure = benchmark_on_simulated_data(
    #         n_features=20, n_tasks=10, n_samples=n_samples
    #     )
    #     benchmarks_n_samples[n_samples] = (duration_cv, duration_sure)

    # joblib.dump(benchmarks_n_samples, "data/simulated_samples_benchmarks.pkl")
    # print("Finished 1st experiment. Saved.")

    # print("\n")
    # print("=" * 15)

    # benchmarks_n_features = {}
    # n_features_to_test = [25, 50, 75, 100, 150, 200]
    # for n_features in tqdm(n_features_to_test, total=len(n_features_to_test)):
    #     duration_cv, duration_sure = benchmark_on_simulated_data(
    #         n_samples=10, n_features=n_features, n_tasks=10
    #     )
    #     benchmarks_n_features[n_features] = (duration_cv, duration_sure)

    # joblib.dump(
    #     benchmarks_n_features, "data/simulated_features_benchmarks.pkl"
    # )
    # print("Finished 2nd experiment. Saved.")

    # print("\n")
    # print("=" * 15)

    # benchmarks_n_tasks = {}
    # n_tasks_to_test = [5, 15, 50, 100, 150]
    # for n_tasks in tqdm(n_tasks_to_test, total=len(n_tasks_to_test)):
    #     duration_cv, duration_sure = benchmark_on_simulated_data(
    #         n_samples=10, n_features=20, n_tasks=n_tasks
    #     )
    #     benchmarks_n_tasks[n_tasks] = (duration_cv, duration_sure)

    # joblib.dump(benchmarks_n_tasks, "data/simulated_tasks_benchmarks.pkl")
    # print("Finished 3rd experiment. Saved.")

    # print("\n")
    # print("=" * 15)

    # Benchmarking on real data
    benchmarks_meg = {}

    for condition in [
        "Left Auditory",
        "Right Auditory",
        "Left visual",
        "Right visual",
    ]:
        duration_sure, duration_cv = benchmark_on_dataset(condition)
        benchmarks_meg[condition] = (duration_sure, duration_cv)

    joblib.dump(benchmarks_meg, "data/meg_benchmarks.pkl")

    print("Finished MEG experiments. Saved.")
