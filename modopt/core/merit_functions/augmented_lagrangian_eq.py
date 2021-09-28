import numpy as np
from modopt.api import MeritFunction


# Note: Augmented Lagrangian is a function of both x and lag_mult
class AugmentedLagrangianEq(MeritFunction):
    def evaluate_function(self, x, lag_mult, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        self.value = obj - np.inner(
            lag_mult, con) + 0.5 * np.inner(rho_vector, con**2)


# Note: Gradient is evaluated with respect to both x and lag_mult

    def evaluate_gradient(self, x, lag_mult, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        grad_x = grad - np.matmul(jac.T, lag_mult - (rho_vector * con))
        grad_lag_mult = -con

        return np.concatenate((grad_x, grad_lag_mult))