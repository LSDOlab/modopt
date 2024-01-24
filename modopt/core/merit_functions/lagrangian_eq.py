import numpy as np
from modopt import MeritFunction


# Note: Lagrangian is a function of both x and lag_mult
class LagrangianEq(MeritFunction):
    def compute_function(self, v):
        nx = self.options['nx']
        x = v[:nx]
        lag_mult = v[nx:]
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj - np.inner(lag_mult, con)

    # Note: Gradient is evaluated with respect to both x and lag_mult
    def compute_gradient(self, v):
        nx = self.options['nx']
        x = v[:nx]
        lag_mult = v[nx:]
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        grad_x = grad - np.matmul(jac.T, lag_mult)
        grad_lag_mult = -con

        return np.concatenate((grad_x, grad_lag_mult))

    def evaluate_function(self, x, lag_mult, f, c):
        return f - np.inner(lag_mult, c)

    def evaluate_gradient(self, x, lag_mult, f, c, g, j):

        grad_x = g - np.matmul(j.T, lag_mult)
        grad_lag_mult = -c

        return np.concatenate((grad_x, grad_lag_mult))