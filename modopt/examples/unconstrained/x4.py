import numpy as np

from modopt.api import Problem


class X4(Problem):
    def initialize(self, ):
        self.problem_name = 'x^4'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(25, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.full((25), 0.1))

        self.add_objective('obj')

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x',
                                        shape=(25, ),
                                        vals=None)
        self.declare_objective_hessian(of='x',
                                       wrt='x',
                                       shape=(25, 25),
                                       vals=None)
        # self.declare_objective_hvp(wrt='x', shape=(25, ), vals=None)

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