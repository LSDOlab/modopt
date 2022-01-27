import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt.api import LineSearch
from scipy.optimize import line_search

# def line_search_wolfe2(f, myfprime, xk, pk, gfk=None, old_fval=None,
#                       old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None,
#                       extra_condition=None, maxiter=10):


class ScipyLS(LineSearch):
    def initialize(self):
        # Wolfe parameter (0.5 for QN methods, 0.9 for Newton-based methods)
        self.options.declare('eta_w',
                             default=0.9,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)
        # Maximum step length
        self.options.declare('max_step',
                             default=1.,
                             types=(int, float),
                             upper=(50.),
                             lower=eps)

        # Maximum number of iterations allowed before convergence
        self.options.declare('max_itr',
                             default=8,
                             types=int,
                             lower=2,
                             upper=100)

    def search(self, x, p, f0=None, g0=None):

        eta_a = self.options['eta_a']
        eta_w = self.options['eta_w']

        if eta_a > eta_w:
            raise ValueError(
                'eta_a should be less than eta_w for existence of positive steps that satisfy strong Wolfe conditions'
            )

        max_itr = self.options['max_itr']
        max_step = self.options['max_step']

        f = self.options['f']
        g = self.options['g']

        num_f_evals = 0
        num_g_evals = 0

        if f0 is None:
            f1 = f(x)
            num_f_evals = 1
        else:
            f1 = f0 * 1.

        if g0 is None:
            g1 = g(x)
            num_g_evals = 1
        else:
            g1 = g0 * 1.

        alpha, num_f_evals, num_g_evals, f2, f1, slope2 = line_search(
            f,
            g,
            x,
            p,
            gfk=g1,
            old_fval=f1,
            old_old_fval=None,
            args=(),
            c1=eta_a,
            c2=eta_w,
            amax=max_step,
            extra_condition=None,
            maxiter=max_itr)

        g2 = "Unavailable"

        converged = True
        if alpha == None:
            converged = False

        return alpha, f2, g2, slope2, num_f_evals, num_g_evals, converged
