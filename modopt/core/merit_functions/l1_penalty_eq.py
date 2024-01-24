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

        return obj + rho * np.linalg.norm(con, 1)

    # Note: Gradient is not continuous, therefore, not useful
    def compute_gradient(self, x):
        rho = self.rho
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        return grad + rho * np.matmul(jac.T, np.sign(con))

    def evaluate_function(self, x, f, c):
        rho = self.rho

        return f + rho * np.linalg.norm(c, 1)

    # Note: Gradient is not continuous, therefore, not useful
    def evaluate_gradient(self, x, f, c, g, j):
        rho = self.rho
        return g + rho * np.matmul(j.T, np.sign(c))
