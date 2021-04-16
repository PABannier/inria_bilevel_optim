import argparse
import joblib
from tqdm import tqdm
import os.path as op

import numpy as np
from numpy.linalg import norm

import mne
from mne.datasets import sample
from mne.viz import plot_sparse_source_estimates
from mne.inverse_sparse.mxne_inverse import _compute_residual

from mtl.mtl import ReweightedMultiTaskLasso
from mtl.cross_validation import ReweightedMultiTaskLassoCV
from mtl.sure import SURE
from mtl.utils_datasets import compute_alpha_max

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alpha",
    help="regularizing hyperparameter",
)
parser.add_argument("--condition", help="condition")
parser.add_argument("--depth", help="depth")

args = parser.parse_args()

if args.condition is None:
    raise ValueError(
        "Please specify a regularizing constant by using --condition argument. "
        + "Available condition: Left Auditory, Right Auditory, Left visual, Right visual."
    )


def load_data():
    data_path = sample.data_path()
    fwd_fname = data_path + "/MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif"
    ave_fname = data_path + "/MEG/sample/sample_audvis-ave.fif"
    cov_fname = data_path + "/MEG/sample/sample_audvis-shrunk-cov.fif"
    trans_fname = op.join(
        data_path, "MEG", "sample", "sample_audvis_raw-trans.fif"
    )
    subjects_dir = data_path + "/subjects"

    bem_fname = op.join(
        subjects_dir, "sample", "bem", "sample-5120-bem-sol.fif"
    )

    condition = args.condition

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # Handling average file
    evoked = mne.read_evokeds(
        ave_fname, condition=condition, baseline=(None, 0)
    )
    evoked.crop(tmin=0.05, tmax=0.15)

    evoked = evoked.pick_types(eeg=False, meg=True)
    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)

    return (
        evoked,
        forward,
        noise_cov,
        cov_fname,
        bem_fname,
        trans_fname,
        subjects_dir,
    )


def apply_solver(solver, evoked, forward, noise_cov, loose=0.2, depth=0.8):
    """Call a custom solver on evoked data.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.

    Parameters
    ----------
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import (
        _prepare_gain,
        is_fixed_orient,
        _reapply_source_weighting,
        _make_sparse_stc,
    )

    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
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

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info["ch_names"]]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)

    n_orient = 1 if is_fixed_orient(forward) else 3
    X, active_set = solver(M, gain, n_orient)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(
        X,
        active_set,
        forward,
        tmin=evoked.times[0],
        tstep=1.0 / evoked.info["sfreq"],
    )

    residual = _compute_residual(forward, evoked, X, active_set, gain_info)

    return stc, residual


def solver(M, G, n_orient=1):
    """Run L2 penalized regression and keep 10 strongest locations.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    if args.alpha:
        estimator = ReweightedMultiTaskLasso(float(args.alpha))
        estimator.fit(G, M)
    else:
        alpha_max = compute_alpha_max(G, M)
        print("Alpha max:", alpha_max)

        alphas = np.geomspace(alpha_max, alpha_max / 10, num=15)
        best_alpha_ = None
        best_sure_ = np.inf

        for alpha in tqdm(alphas, total=len(alphas)):
            criterion = SURE(ReweightedMultiTaskLasso, 1, random_state=0)
            sure_val_ = criterion.get_val(G, M, alpha)
            if sure_val_ < best_sure_:
                best_sure_ = sure_val_
                best_alpha_ = alpha

        print("best sure", best_sure_)

        # Refitting
        estimator = ReweightedMultiTaskLasso(best_alpha_)
        estimator.fit(G, M)

    X = estimator.coef_
    active_set = norm(X, axis=1) != 0

    return X[active_set, :], active_set


if __name__ == "__main__":
    loose, depth = 0, args.depth if args.depth else 0.9
    (
        evoked,
        forward,
        noise_cov,
        cov_fname,
        bem_fname,
        trans_fname,
        subjects_dir,
    ) = load_data()

    stc, residual = apply_solver(
        solver, evoked, forward, noise_cov, loose, depth
    )

    print("=" * 10)
    print("Explained variance:", norm(residual.data) / norm(evoked.data))
    print("=" * 10)

    plot_sparse_source_estimates(
        forward["src"], stc, bgcolor=(1, 1, 1), opacity=0.1
    )
