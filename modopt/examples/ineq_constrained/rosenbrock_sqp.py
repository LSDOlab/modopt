import numpy as np

from modopt import Problem


class Rosenbrock(Problem):
    # a = 1
    # b = 100

    def initialize(self):
        self.problem_name = 'rosenbrock'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.array([-1.2, 1.]))

        self.add_constraints('x^2 + y^2',
                             shape=(1, ),
                             lower=None,
                             upper=None,
                             equals=np.array([
                                 2,
                             ]))

        self.name_objective('obj')

    def compute_objective(self, x):
        f = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        return f

    def compute_objective_gradient(self, x):
        grad = np.array([
            -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1),
            200 * (x[1] - x[0]**2)
        ])
        return grad

    def compute_constraints(self, x):
        c = x[0]**2 + x[1]**2
        return np.array([
            c,
        ])

    def compute_constraint_jacobian(self, x):
        jac = np.array([[2 * x[0], 2 * x[1]]])
        return jac.reshape((1, jac.size))

    def compute_objective_hessian(self, x):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])

        return hess

    def compute_objective_hvp(self, x, p):
        hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
                         [-400 * x[0], 200]])
        hvp = np.matmul(hess, p)

        return hvp