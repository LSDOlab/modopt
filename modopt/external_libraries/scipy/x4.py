import numpy as np

from modopt import Problem


class X4(Problem):
    def initialize(self, ):
        # self.options.declare('problem_name', default='x^4', types=str)
        self.problem_name = 'x^4'

    def compute_objective(self, x):
        x4 = np.power(x, 4)
        f = np.sum(x4)
        return f

    def compute_objective_gradient(self, x):
        grad = 4 * np.power(x, 3)
        return grad

    def compute_objective_hessian(self, x):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        return hess

    def compute_objective_hvp(self, x, p):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        hvp = np.matmul(hess, p)

        return hvp