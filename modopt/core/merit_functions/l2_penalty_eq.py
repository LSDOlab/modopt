import numpy as np
from modopt.api import MeritFunction


class L2Eq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        self.value = obj + 0.5 * np.inner(rho_vector, con**2)

    def evaluate_gradient(self, x, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        return grad + np.matmul(jac.T, (rho_vector * con))
