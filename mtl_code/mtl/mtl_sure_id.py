import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import MultiTaskLasso
from mtl.simulated_data import simulate_data
from mtl.utils_datasets import compute_alpha_max


def build_jacob_prox_z(Z, supp, alpha):
    # Z : array, shape (n_supp, n_times)
    n_times = Z.shape[1]
    jacob_prox_z = np.zeros((supp.sum(), n_times, n_times))
    for j in range(supp.sum()):
        z_j = Z[j, :]
        nrm = norm(z_j)
        if nrm <= alpha:
            jacob_prox_z[j, :, :] = 0
        else:
            jacob_prox_z[j, :, :] = ((1 - alpha / nrm) * np.eye(n_times) 
                                     + (alpha / nrm ** 3) * np.outer(z_j, z_j))
    return jacob_prox_z



n_samples = 10
n_features = 30
n_tasks = 10

X, Y = simulate_data(n_samples, n_features, n_tasks, nnz=3, random_state=0)[:2]
alpha_max = compute_alpha_max(X, Y)
alpha = alpha_max * 0.1

L = norm(X.T @ X, ord=2)

# Closed-form solution: dof = support size
clf = MultiTaskLasso(alpha, fit_intercept=False)
clf.fit(X, Y)
supp = norm(clf.coef_.T, axis=-1) != 0
dof_cf = n_tasks * np.sum(supp)

# Analytical solution via implicit differentiation
X_supp = X[:, supp]
B = clf.coef_.T[supp, :]
Z = B - X_supp.T @ (X_supp @ B - Y) / L
jacob_prox_z = build_jacob_prox_z(Z, supp, alpha)

id_feat = np.eye(supp.sum())
import ipdb; ipdb.set_trace()
J = np.linalg.inv(id_feat - jacob_prox_z * (id_feat - X_supp.T @ X_supp / L)) @ (jacob_prox_z * X_supp.T / L)
dof_id = np.trace(X_supp @ J)

print("DOF")
print("Closed form:", dof_cf)
print("Implicit differentiation:", dof_id)
