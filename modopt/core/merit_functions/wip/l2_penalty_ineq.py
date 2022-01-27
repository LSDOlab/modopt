import numpy as np
from modopt.api import MeritFunction


# Note: This Merit function is for problems in all-inequality form, i.e., c(x) >= 0
class L2Ineq(MeritFunction):
    def setup(self):
        nc = self.options['nc']
        self.rho = np.zeros(nc)

    def set_rho(self, rho):
        nc = self.options['nc']
        if type(rho) in (np.int32, np.int64, int, np.float64, float):
            self.rho = np.ones(nc) * rho
        else:
            self.rho = rho * 1.

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        con_violations = np.maximum(-con, 0)
        # violated_constraints = np.where(con<0)[0]

        return obj + 0.5 * np.inner(rho, con_violations**2)
        # return obj + 0.5 * np.inner(rho_vector[violated_constraints], con[violated_constraints] ** 2)

    def evaluate_function(self, x, f, c):
        rho = self.rho
        con_violations = np.maximum(-c, 0)

        return f + 0.5 * np.inner(rho, con_violations**2)

    def compute_gradient(self, x):
        rho = self.rho
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        con_violations = np.maximum(-con, 0)
        violated_constraints = np.where(con < 0)[0]

        # return grad + np.matmul(-jac.T, (rho * con_violations))

        return grad + np.matmul(
            -jac[violated_constraints].T,
            (rho * con_violations)[violated_constraints])

    def evaluate_gradient(self, x, f, c, g, j):
        rho = self.rho

        con_violations = np.maximum(-c, 0)
        violated_constraints = np.where(c < 0)[0]

        # return grad + np.matmul(-jac.T, (rho * con_violations))

        return g + np.matmul(
            -j[violated_constraints].T,
            (rho * con_violations)[violated_constraints])