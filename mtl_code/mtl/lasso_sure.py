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
alpha = alpha_max * 0.1

L = norm(X.T @ X, ord=2)

# Closed-form solution: dof = support size
clf = Lasso(alpha, fit_intercept=False)
clf.fit(X, y)
supp = clf.coef_ != 0
dof_cf = np.sum(supp != 0)

# Analytical solution via implicit differentiation
# beta = clf.coef_
# z = beta - (1 / L) * X.T @ (X @ beta - y)
# supp = np.abs(z) >= alpha
jacob_prox_z = np.ones(supp.sum())  # to be modified for other penalties

id_feat = np.eye(supp.sum())
X_supp = X[:, supp]
J = np.linalg.inv(id_feat - jacob_prox_z * (id_feat - X_supp.T @ X_supp / L)) @ (jacob_prox_z[:, None] * X_supp.T / L)
dof_id = np.trace(X_supp @ J)

print("DOF")
print("Closed form:", dof_cf)
print("Implicit differentiation:", dof_id)
