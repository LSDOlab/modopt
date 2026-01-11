import numpy as np
from modopt import MeritFunction


class L1Eq(MeritFunction):
    def setup(self):
        self.rho = 0.

    def set_rho(self, rho):
        self.rho = rho * 1.

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + np.linalg.norm(rho * con, 1)

    # Note: Gradient is not continuous, therefore, not useful
    def compute_gradient(self, x):
        rho = self.rho
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        return grad + np.matmul(jac.T, rho * np.sign(con))

    def evaluate_function(self, x, f, c):
        rho = self.rho

        return f + np.linalg.norm(rho * c, 1)

    # Note: Gradient is not continuous, therefore, not useful
    def evaluate_gradient(self, x, f, c, g, j):
        rho = self.rho
        return g + np.matmul(j.T, rho * np.sign(c))
