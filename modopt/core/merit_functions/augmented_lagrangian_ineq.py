import numpy as np
from modopt.api import MeritFunction


# Note: Augmented Lagrangian is a function of  x, lag_mult and slack variables slacks
# Note: This Merit function is for problems in all-inequality form, i.e., c(x) >= 0
class AugmentedLagrangianIneq(MeritFunction):
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
        nc = self.options['nc']
        rho = self.rho
        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = v[(nx + nc):]
        obj = self.options['f'](x)
        con = self.options['c'](x)

        return obj - np.dot(
            lag_mult,
            (con - slacks)) + 0.5 * np.inner(rho, (con - slacks)**2)

    def evaluate_function(self, x, lag_mult, s, f, c):
        rho = self.rho

        return f - np.dot(lag_mult,
                          (c - s)) + 0.5 * np.inner(rho, (c - s)**2)


# Note: Gradient is evaluated with respect to x, lag_mult and slack variables slacks

    def compute_gradient(self, v):
        nx = self.options['nx']
        nc = self.options['nc']
        rho = self.rho
        x = v[:nx]
        lag_mult = v[nx:(nx + nc)]
        slacks = v[(nx + nc):]
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        # grad_x = grad - np.matmul(jac.T, lag_mult - (rho *
        #                                              (con - slacks)))
        grad_x = grad - jac.T @ (lag_mult - (rho * (con - slacks)))
        grad_lag_mult = -(con - slacks)
        grad_slacks = lag_mult - rho * (con - slacks)

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))

    def evaluate_gradient(self, x, lag_mult, s, f, c, g, j):
        rho = self.rho

        # grad_x = g - np.matmul(j.T, lag_mult - (rho * (c - s)))
        grad_x = g - j.T @ (lag_mult - (rho * (c - s)))
        grad_lag_mult = -(c - s)
        grad_slacks = lag_mult - rho * (c - s)

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))