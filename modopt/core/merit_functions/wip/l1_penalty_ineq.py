import numpy as np
from modopt import MeritFunction


class L1Ineq(MeritFunction):
    def setup(self):
        self.rho = 0.

    def set_rho(self, rho):
        self.rho = rho * 1.

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        con_violations = np.maximum(-con, 0)

        return obj + rho * np.sum(con_violations)

    def evaluate_function(self, x, f, c):
        rho = self.rho
        con_violations = np.maximum(-c, 0)

        return f + rho * np.sum(con_violations)

    def compute_gradient(self, x):
        rho = self.rho
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        con_violations = np.maximum(-con, 0)

        return grad + rho * np.matmul(jac.T, -np.sign(
            con_violations))  # Note: constraint violations >= 0
