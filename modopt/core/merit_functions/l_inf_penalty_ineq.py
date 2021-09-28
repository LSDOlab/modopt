import numpy as np
from modopt.api import MeritFunction


class LInfIneq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        con_violations = np.maximum(-con, 0)

        return obj + rho * np.max(con_violations)

    def evaluate_gradient(self, x, g, j, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        con_violations = np.maximum(-con, 0)

        # Note: only taking the first maximum of constraint violations
        max_location = np.where(np.max(np.abs(con_violations)))[0][0]

        return grad + rho * (-1) * jac[max_location]
