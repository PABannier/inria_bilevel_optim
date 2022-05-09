import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso
from mtl.simulated_data import simulate_data


n_samples = 100
n_features = 1000
n_tasks = 1

X, y = simulate_data(n_samples, n_features, n_tasks, nnz=3, random_state=0)[:2]
y = np.ravel(y)
alpha_max = norm(X.T @ y, ord=np.inf) / len(X)
alpha = alpha_max * 0.5

L = norm(X.T @ X, ord=2)

# Closed-form solution: dof = support size
clf = Lasso(alpha, fit_intercept=False)
clf.fit(X, y)
dof_cf = np.sum(clf.coef_ != 0)

# Analytical solution via implicit differentiation
beta = clf.coef_
z = beta - (1 / L) * X.T @ (X @ beta - y)
jacob_prox_z = np.zeros_like(z)
jacob_prox_z[np.abs(z) >= alpha] = 1
jacob_prox_z = np.diag(jacob_prox_z)

id_feat = np.eye(n_features)
J = np.linalg.inv(id_feat - jacob_prox_z @ (id_feat - X.T @ X / L)) @ jacob_prox_z @ X.T / L
dof_id = np.trace(X @ J)

print("DOF")
print("Closed form:", dof_cf)
print("Implicit differentiation:", dof_id)
