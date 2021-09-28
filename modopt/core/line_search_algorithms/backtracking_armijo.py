import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt.api import LineSearch


class BacktrackingArmijo(LineSearch):
    def initialize(self):
        # Stepsize contraction factor
        self.options.declare('gamma_c',
                             default=0.6,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)

        # Maximum number of iterations allowed before convergence
        self.options.declare('max_it', default=20, types=int, lower=3)

    def search(self, x, p, f0=None, g0=None):

        eta_a = self.options['eta_a']
        gamma_c = self.options['gamma_c']
        max_it = self.options['max_it']
        f = self.options['f']
        g = self.options['g']

        alpha = 1.
        num_f_evals = 0
        num_g_evals = 0

        if f0 is None:
            f1 = f(x)
            num_f_evals = 1
        else:
            f1 = f0 * 1.

        if g0 is None:
            g1 = g(x)
            g_evals = 1
        else:
            g1 = g0 * 1.

        slope = np.inner(g1, p)

        itr = 0
        rho = 0.
        while (itr <= max_it) and (rho < eta_a):
            f2 = f(x + alpha * p)
            rho = (f1 - f2) / (alpha * slope)
            alpha *= gamma_c
            itr += 1

        num_f_evals += itr

        if rho < eta_a:
            converged = False

        return alpha, f2, num_f_evals, num_g_evals, converged