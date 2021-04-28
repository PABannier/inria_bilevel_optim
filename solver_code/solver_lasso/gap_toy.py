import numpy as np

from mtl.simulated_data import simulate_data


def sum_squared(X):
    """Computes the Frobenius norm for a matrix
    in a more efficient way.
    """
    X_flat = X.ravel(order="F" if np.isfortran(X) else "C")
    return np.dot(X_flat, X_flat)


def group_norm_2(A, n_orient):
    """Computes a group norm (l2-norm over the columns)
    taking into account the dipole orientiation.
    """
    n_positions = A.shape[0] // n_orient
    return np.sum(np.power(A, 2, A).reshape(n_positions, -1), axis=1)


def norm_l_2_1(A, n_orient):
    """Mixed l_2_1 norm for matrix"""
    return np.sum(np.sqrt(group_norm_2(A, n_orient)))


def primal(X, Y, W, alpha, n_orient):
    R = X @ W - Y
    p_obj = sum_squared(R) / (2 * np.prod(X.shape))
    p_obj += alpha * norm_l_2_1(W, n_orient)
    return p_obj


if __name__ == "__main__":
    X, Y, W, _ = simulate_data()

    p_obj = primal(X, Y, W, 0.3, 1)
    print(p_obj)
