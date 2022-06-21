import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import MultiTaskLasso
from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max


def build_jacob_prox_z(Z, alpha):
    """Builds the Jacobian matrix of the proximal operator of z w.r.t. z

    Parameters
    ----------
    Z : array, shape (supp_size, n_times)
        Input matrix.

    alpha : float
        Threshold parameter for the proximal operator.

    Returns
    -------
    jacobian : array, shape (supp_size * n_times, supp_size * n_times)
        Block-diagonal matrix containing sub-Jacobian matrices.
    """
    supp_size, n_times = Z.shape
    jacobian = np.zeros((supp_size * n_times, supp_size * n_times))
    for j in range(supp_size):
        z_j = Z[j, :]
        nrm = norm(z_j)
        sub_jacobian = ((1 - alpha / nrm) * np.eye(n_times)
                         + (alpha / nrm ** 3) * np.outer(z_j, z_j))
        low_idx = j * n_times
        up_idx = (j + 1) * n_times
        jacobian[low_idx : up_idx, low_idx : up_idx] = sub_jacobian
    return jacobian


def build_XtX(X, n_tasks):
    n_features = X.shape[1]
    XtX_enlarged = np.zeros((n_features * n_tasks, n_features * n_tasks))
    for t in range(n_tasks):
        low_idx = t * n_features
        up_idx = (t + 1) * n_features
        XtX_enlarged[low_idx : up_idx, low_idx : up_idx] = X.T @ X
    return XtX_enlarged


n_samples = 5
n_features = 7
n_tasks = 4

X, Y = simulate_data(n_samples, n_features, n_tasks, nnz=3, random_state=0)[:2]
alpha_max = compute_alpha_max(X, Y)
alpha = alpha_max * 0.1

L = norm(X.T @ X, ord=2)

# Closed-form solution: dof = support size
clf = MultiTaskLasso(alpha, fit_intercept=False)
clf.fit(X, Y)
supp = norm(clf.coef_.T, axis=-1) != 0
supp_size = np.sum(supp)
dof_cf = n_tasks * supp_size

# Analytical solution via implicit differentiation
X_s = X[:, supp]
B_s = clf.coef_.T[supp, :]
Z_s = B_s - X_s.T @ (X_s @ B_s - Y) / L  # (supp_size, n_times)
# z_s = Z_s.reshape((supp_size * n_tasks))  # (supp_size * n_times)

jacob_prox_Z_s = build_jacob_prox_z(Z_s, alpha)  # (supp_size * n_times, supp_size * n_times)

# X_enlarged = np.tile(X_s, (1, n_tasks))  # (n_samples, supp_size * n_times)
X_enlarged = np.repeat(X_s[:, :, np.newaxis], n_tasks, axis=2)
X_enlarged = np.reshape(X_enlarged, (n_samples, supp_size * n_tasks))

XtX_enlarged = build_XtX(X_s, n_tasks)  # block diagonal (supp_size * n_times, supp_size * n_times)
Id = np.eye(supp_size * n_tasks)  # (supp_size * n_times, supp_size * n_times)

J = np.linalg.inv(Id - jacob_prox_Z_s @ (Id - XtX_enlarged / L)) @ jacob_prox_Z_s @ X_enlarged.T / L
dof_id = np.trace(X_enlarged @ J)

print("DOF")
print("Closed form:", dof_cf)
print("Implicit differentiation:", dof_id)




# Analytical solution via implicit differentiation
# X_supp = X[:, supp]
# B = clf.coef_.T[supp, :]
# Z = B - X_supp.T @ (X_supp @ B - Y) / L
# jacob_prox_z = build_jacob_prox_z(Z, supp, alpha)

# id_feat = np.eye(supp.sum())
# J = np.linalg.inv(id_feat - jacob_prox_z * (id_feat - X_supp.T @ X_supp / L)) @ (jacob_prox_z * X_supp.T / L)
# dof_id = np.trace(X_supp @ J)
