import numpy as np
from modopt.api import MeritFunction


# Note: Modified Lagrangian is a function of only x
#       (but depends on x_k, pi_k)
class ModifiedLagrangianIneq(MeritFunction):
    def setup(self):
        nx = self.options['nx']
        self.x_k = np.zeros(nx)

    def set_x_k(self, x_k, c_k=None, J_k=None):
        nx = self.options['nx']
        if len(x_k) != nx:
            raise ValueError("Size of x_k provided does not match nx")

        self.x_k = x_k * 1.
        self.c_k = c_k * 1.
        self.J_k = J_k * 1.
        if c_k is None:
            self.c_k = self.options['c'](x_k)
        if J_k is None:
            self.J_k = self.options['j'](x_k)

    def set_pi_k(self, pi_k):
        nc = self.options['nc']
        if len(pi_k) != nc:
            raise ValueError("Size of pi_k provided does not match nc")

        self.pi_k = pi_k * 1.

    def compute_function(self, x):
        x_k = self.x_k
        pi_k = self.pi_k
        c_k = self.c_k
        J_k = self.J_k

        obj = self.options['f'](x)
        con = self.options['c'](x)

        # Departure from linearity
        dl = con - (c_k + J_k @ (x - x_k))

        return obj - np.inner(pi_k, dl)

    def evaluate_function(self, x, f, c):
        x_k = self.x_k
        pi_k = self.pi_k
        c_k = self.c_k
        J_k = self.J_k

        # Departure from linearity
        dl = c - (c_k + J_k @ (x - x_k))

        return f - np.inner(pi_k, dl)

    # Note: Gradient is evaluated with respect to only x
    def compute_gradient(self, x):
        pi_k = self.pi_k
        J_k = self.J_k

        grad = self.options['g'](x)
        jac = self.options['j'](x)

        return grad - (jac - J_k).T @ pi_k

    def evaluate_gradient(self, x, f, c, g, j):
        pi_k = self.pi_k
        J_k = self.J_k

        return g - (j - J_k).T @ pi_k