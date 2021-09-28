import numpy as np
from modopt.api import MeritFunction


class L1Eq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + rho * np.linalg.norm(con, 1)

    def evaluate_gradient(self, x, g, j, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        return grad + rho * np.matmul(jac.T, np.sign(con))
