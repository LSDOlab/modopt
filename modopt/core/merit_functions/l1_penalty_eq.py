import numpy as np
from modopt import MeritFunction


class L1Eq(MeritFunction):
    def setup(self):
        nc = self.options['nc']
        self.rho = np.zeros(nc)

    def set_rho(self, rho):
        self.rho[:] = rho

    def compute_function(self, x):
        rho = self.rho
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj + np.linalg.norm(rho * con, 1)

    def compute_gradient(self, x): # Note: Gradient is not continuous
        rho  = self.rho
        con  = self.options['c'](x)
        grad = self.options['g'](x)
        jac  = self.options['j'](x)

        return grad + np.matmul(jac.T, rho * np.sign(con))

    def evaluate_function(self, x, f, c):
        rho = self.rho

        return f + np.linalg.norm(rho * c, 1)

    def evaluate_gradient(self, x, f, c, g, j): # Note: Gradient is not continuous
        rho = self.rho
        
        return g + np.matmul(j.T, rho * np.sign(c))