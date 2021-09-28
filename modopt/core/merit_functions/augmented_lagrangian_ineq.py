import numpy as np
from modopt.api import MeritFunction


# Note: Augmented Lagrangian is a function of  x, lag_mult and slack variables slacks
# Note: This Merit function is for problems in all-inequality form, i.e., c(x) >= 0
class AugmentedLagrangianIneq(MeritFunction):
    def evaluate_function(self, x, lag_mult, slacks, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        self.value = obj - np.inner(
            lag_mult,
            (con - slacks)) + 0.5 * np.inner(rho_vector,
                                             (con - slacks)**2)


# Note: Gradient is evaluated with respect to x, lag_mult and slack variables slacks

    def evaluate_gradient(self, x, lag_mult, slacks, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        grad_x = grad - np.matmul(
            jac.T, lag_mult - (rho_vector * (con - slacks)))
        grad_lag_mult = -(con - slacks)
        grad_slacks = lag_mult - rho * (con - slacks)

        return np.concatenate((grad_x, grad_lag_mult, grad_slacks))
