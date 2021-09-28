import numpy as np
from modopt.api import MeritFunction


class LInfEq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + rho * np.linalg.norm(con, np.inf)

    def evaluate_gradient(self, x, g, j, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        # Note: only taking the first maximum
        max_location = np.where(np.max(np.abs(con)))[0][0]

        return grad + rho * np.sign(
            con[max_location]) * jac[max_location]
