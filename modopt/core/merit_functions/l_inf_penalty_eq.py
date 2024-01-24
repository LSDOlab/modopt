import numpy as np
from modopt import MeritFunction


class LInfEq(MeritFunction):
    def setup(self):
        self.rho = 0.

    def set_rho(self, rho):
        self.rho = rho * 1.

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + rho * np.linalg.norm(con, np.inf)

    # Note: Gradient is not continuous, therefore, not useful
    def compute_gradient(self, x):
        rho = self.rho
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        # Note: only taking the first maximum
        abs_c = np.abs(con)
        abs_max = np.max(abs_c)
        max_location = np.where(abs_c == abs_max)[0][0]

        return grad + rho * np.sign(
            con[max_location]) * jac[max_location]

    def evaluate_function(self, x, f, c):
        rho = self.rho

        return f + rho * np.linalg.norm(c, np.inf)

    # Note: Gradient is not continuous, therefore, not useful
    def evaluate_gradient(self, x, f, c, g, j):
        rho = self.rho

        # Note: only taking the first maximum
        abs_c = np.abs(c)
        abs_max = np.max(abs_c)
        max_location = np.where(abs_c == abs_max)[0][0]

        return g + rho * np.sign(c[max_location]) * j[max_location]
