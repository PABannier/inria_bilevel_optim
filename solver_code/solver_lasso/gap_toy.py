import numpy as np
from numpy.linalg import norm

from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max


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


def _bcd(
    G,
    X,
    R,
    active_set,
    one_ovr_lc,
    n_orient,
    n_positions,
    alpha_lc,
    list_G_j_c,
):
    X_j_new = np.zeros_like(X[0:n_orient, :], order="C")

    for j, G_j_c in enumerate(list_G_j_c):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        G_j = G[:, idx]
        X_j = X[idx]

        X_j_new = G_j.T @ R * one_ovr_lc[j]

        # Mathurin's trick to avoid checking all the entries
        was_non_zero = X_j[0, 0] != 0
        # was_non_zero = np.any(X_j)

        if was_non_zero:
            R += np.dot(G_j, X_j)
            X_j_new += X_j
        block_norm = np.sqrt(sum_squared(X_j_new))
        if block_norm <= alpha_lc[j]:
            X_j.fill(0.0)
            active_set[idx] = False
        else:
            shrink = max(1.0 - alpha_lc[j] / block_norm, 0.0)
            X_j_new *= shrink
            R -= np.dot(G_j, X_j_new)
            X_j[:] = X_j_new
            active_set[idx] = True


def _mixed_norm_solver_bcd(
    M,
    G,
    alpha,
    lipschitz_constant,
    maxit=200,
    tol=1e-8,
    verbose=None,
    init=None,
    n_orient=1,
    dgap_freq=10,
):
    """Solve L21 inverse problem with block coordinate descent."""
    n_sensors, n_times = M.shape
    n_sensors, n_sources = G.shape
    n_positions = n_sources // n_orient

    if init is None:
        X = np.zeros((n_sources, n_times))
        R = M.copy()
    else:
        X = init
        R = M - np.dot(G, X)

    E = []  # track primal objective function
    highest_d_obj = -np.inf
    active_set = np.zeros(n_sources, dtype=bool)  # start with full AS

    alpha_lc = alpha / lipschitz_constant

    # First make G fortran for faster access to blocks of columns
    G = np.asfortranarray(G)
    # Ensure these are correct for dgemm
    assert R.dtype == np.float64
    assert G.dtype == np.float64
    one_ovr_lc = 1.0 / lipschitz_constant

    # assert that all the multiplied matrices are fortran contiguous
    assert X.T.flags.f_contiguous
    assert R.T.flags.f_contiguous
    assert G.flags.f_contiguous
    # storing list of contiguous arrays
    list_G_j_c = []
    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        list_G_j_c.append(np.ascontiguousarray(G[:, idx]))

    for i in range(maxit):
        _bcd(
            G,
            X,
            R,
            active_set,
            one_ovr_lc,
            n_orient,
            n_positions,
            alpha_lc,
            list_G_j_c,
        )

        if (i + 1) % dgap_freq == 0:
            _, p_obj, d_obj, _ = dgap_l21(
                M, G, X[active_set], active_set, alpha, n_orient
            )
            highest_d_obj = max(d_obj, highest_d_obj)
            gap = p_obj - highest_d_obj
            E.append(p_obj)
            print(
                "Iteration %d :: p_obj %f :: dgap %f :: n_active %d"
                % (i + 1, p_obj, gap, np.sum(active_set) / n_orient)
            )

            if gap < tol:
                print("Convergence reached ! (gap: %s < %s)" % (gap, tol))
                break

    X = X[active_set]

    return X, active_set, E


if __name__ == "__main__":
    X, Y, W, _ = simulate_data(
        n_samples=10, n_features=15, n_tasks=10, nnz=4, random_state=0
    )
    n_orient = 1

    n_positions = X.shape[1] // n_orient

    # Alpha max
    alpha_max = compute_alpha_max(X, Y)

    # Compute Lipschitz constant
    lipschitz_constant = np.empty((n_positions))

    for j in range(n_positions):
        idx = slice(j * n_orient, (j + 1) * n_orient)
        lipschitz_constant[j] = norm(X[:, idx].T @ X[:, idx])

    # Call solver
    _mixed_norm_solver_bcd(
        Y, X, alpha_max / 10, lipschitz_constant, n_orient=n_orient, maxit=200
    )
