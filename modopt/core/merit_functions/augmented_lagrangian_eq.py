import numpy as np
from modopt.api import MeritFunction


# Note: Augmented Lagrangian is a function of both x and lag_mult
class AugmentedLagrangianEq(MeritFunction):
    def setup(self):
        nc = self.options['nc']
        self.rho = np.zeros(nc)

    def set_rho(self, rho):
        nc = self.options['nc']
        if type(rho) in (np.int32, np.int64, int, np.float64, float):
            self.rho = np.ones(nc) * rho
        else:
            self.rho = rho * 1.

    def compute_function(self, v):
        nx = self.options['nx']
        rho = self.rho
        x = v[:nx]
        lag_mult = v[nx:]
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj - np.inner(lag_mult,
                              con) + 0.5 * np.inner(rho, con**2)


# Note: Gradient is evaluated with respect to both x and lag_mult

    def compute_gradient(self, v):
        nx = self.options['nx']
        rho = self.rho
        x = v[:nx]
        lag_mult = v[nx:]
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        grad_x = grad - np.matmul(jac.T, lag_mult - (rho * con))
        grad_lag_mult = -con

        return np.concatenate((grad_x, grad_lag_mult))

    def evaluate_function(self, x, lag_mult, f, c):
        rho = self.rho

        return f - np.inner(lag_mult, c) + 0.5 * np.inner(rho, c**2)

    def evaluate_gradient(self, x, lag_mult, f, c, g, j):
        rho = self.rho

        grad_x = g - np.matmul(j.T, lag_mult - (rho * c))
        grad_lag_mult = -c

        return np.concatenate((grad_x, grad_lag_mult))