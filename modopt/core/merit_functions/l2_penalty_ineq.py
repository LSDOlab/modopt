import numpy as np
from modopt.api import MeritFunction


# Note: This Merit function is for problems in all-inequality form, i.e., c(x) >= 0
class L2Ineq(MeritFunction):
    def evaluate_function(self, x, rho):
        obj = self.options['f'](x)
        con = self.options['c'](x)

        con_violations = np.maximum(-con, 0)
        # violated_constraints = np.where(con<0)[0]

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        self.value = obj + 0.5 * np.inner(rho_vector, con_violations**2)
        # self.value = obj + 0.5 * np.inner(rho_vector[violated_constraints], con[violated_constraints] ** 2)

    def evaluate_gradient(self, x, rho):
        con = self.options['c'](x)
        grad = self.options['g'](x)
        jac = self.options['j'](x)

        con_violations = np.maximum(-con, 0)
        violated_constraints = np.where(con < 0)[0]

        rho_vector = rho * 1.
        if type(rho) in (int, float):
            rho_vector = np.ones(len(con)) * rho

        # return grad + np.matmul(jac.T, (rho_vector * con_violations))

        return grad + np.matmul(
            jac[violated_constraints].T,
            (rho_vector * con_violations)[violated_constraints])

        # return grad + np.matmul(jac[violated_constraints].T, (rho_vector[violated_constraints] * con[violated_constraints]))
