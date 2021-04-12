import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.utils import check_random_state

from solver_lasso.cd import CDLasso
from solver_lasso.ista import ProxLasso
from solver_lasso.utils import compute_alpha_max

X, y = load_boston(return_X_y=True)

alpha_max = compute_alpha_max(X, y)
alpha = alpha_max * 0.01


cd_clf = CDLasso(alpha, accelerated=False)
cd_clf_accelerated = CDLasso(alpha, accelerated=True)
ista_clf = ProxLasso(alpha, accelerated=False)
fista_clf = ProxLasso(alpha, accelerated=True)

cd_clf.fit(X, y)
cd_clf_accelerated.fit(X, y)
ista_clf.fit(X, y)
fista_clf.fit(X, y)

fig = plt.figure()

plt.plot(np.log(cd_clf.gap_history_), label="CD", color="orange")
plt.plot(
    np.log(cd_clf_accelerated.gap_history_), label="Anderson-CD", color="blue"
)
plt.plot(np.log(ista_clf.gap_history_), label="ISTA", color="black")
plt.plot(np.log(fista_clf.gap_history_), label="FISTA", color="green")

plt.legend()

plt.xlabel("Iteration")
plt.ylabel("$p^* - d^*$ (logarithmic)")

plt.title("Convergence speed", fontsize=13)

plt.show(block=True)