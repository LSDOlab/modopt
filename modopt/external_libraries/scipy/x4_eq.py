import numpy as np
import scipy.sparse as sp

from modopt import Problem


class X4(Problem):
    def initialize(self):
        self.problem_name = 'x^4'

    def evaluate_objective(self, x):
        x4 = np.power(x, 4)
        f = np.sum(x4)
        return f

    def evaluate_constraints_and_residuals(self, x):
        x3 = np.power(x, 3) - 1
        return x3

    def compute_gradient(self, x):
        grad = 4 * np.power(x, 3)
        return grad

    def compute_constraint_and_residual_jacobian(self, x):
        jac = 3 * np.power(x, 2)
        return np.diag(jac)

    def compute_hessian(self, x):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        return hess

    def compute_hvp(self, x, dir):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        hvp = np.matmul(hess, p)

        return hvp