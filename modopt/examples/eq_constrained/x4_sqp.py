import numpy as np
import scipy.sparse as sp

from modopt import Problem


class X4(Problem):
    def initialize(self):
        self.problem_name = 'x^4'

    def setup(self):
        # Make optional
        self.add_design_variables('x',
                                  shape=(1000, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.full((1000, ), 5.))

        # self.name_objective('obj')

        self.add_constraints('x^3 >= 1',
                             shape=(1000, ),
                             lower=np.ones((1000, )),
                             upper=None,
                             equals=None)

    def setup_derivatives(self):
        pass

    def compute_objective(self, x):
        # f = np.sum(x**4)
        f = np.sum(np.power(x, 4))
        return f

    def compute_objective_gradient(self, x):
        # grad = 4 * x**3
        grad = 4 * np.power(x, 3)
        return grad

    def compute_constraints(self, x):
        # x3 = x**3
        x3 = np.power(x, 3)
        return x3

    def compute_constraint_jacobian(self, x):
        # jac = 3 * x**2
        jac = 3 * np.power(x, 2)
        return np.diag(jac)

    def compute_objective_hessian(self, x):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        return hess

    def compute_objective_hvp(self, x, dir):
        nx = x.shape[0]
        diag = 12 * np.power(x, 2).reshape(nx, )
        hess = np.diag(diag)
        hvp = np.matmul(hess, p)

        return hvp