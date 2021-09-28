import numpy as np
from modopt.api import MeritFunction


class L1Ineq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        con_violations = np.maximum(-con, 0)

        return obj + rho * np.sum(con_violations)

    def evaluate_gradient(self, x, g, j, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        con_violations = np.maximum(-con, 0)

        return grad + rho * np.matmul(jac.T, -np.sign(
            con_violations))  # Note: constraint violations >= 0
