import numpy as np
from modopt import MeritFunction


class L2Eq(MeritFunction):
    def setup(self):
        nc = self.options['nc']
        self.rho = np.zeros(nc)

    def set_rho(self, rho):
        self.rho[:] = rho

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + 0.5 * np.inner(rho, con**2)

    def compute_gradient(self, x):
        rho  = self.rho
        con  = self.options['c'](x)
        grad = self.options['g'](x)
        jac  = self.options['j'](x)

        return grad + np.matmul(jac.T, (rho * con))

    def evaluate_function(self, x, f, c):
        rho = self.rho

        return f + 0.5 * np.inner(rho, c**2)

    def evaluate_gradient(self, x, f, c, g, j):
        rho = self.rho
        
        return g + np.matmul(j.T, (rho * c))