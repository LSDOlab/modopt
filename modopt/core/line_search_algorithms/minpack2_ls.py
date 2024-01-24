import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt import LineSearch
from scipy.optimize.linesearch import line_search_wolfe1 as line_search

# def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
#                        old_fval=None, old_old_fval=None,
#                        args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
#                        xtol=1e-14):


class Minpack2LS(LineSearch):
    def initialize(self):
        # Wolfe parameter (0.5 for QN methods, 0.9 for Newton-based methods)
        self.options.declare('eta_w',
                             default=0.9,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)
        # Maximum step length
        self.options.declare('max_step',
                             default=50,
                             types=(int, float),
                             upper=(1e10),
                             lower=eps)
        # Minimum step length
        self.options.declare('min_step',
                             default=1e-8,
                             types=(int, float),
                             upper=(1.0 - eps),
                             lower=eps)

        # Relative tolerance for an acceptable step
        self.options.declare('alpha_tol',
                             default=1e-14,
                             types=float,
                             upper=(1.0 - eps),
                             lower=eps)

    def search(self, x, p, f0=None, g0=None):

        eta_a = self.options['eta_a']
        eta_w = self.options['eta_w']

        if eta_a > eta_w:
            raise ValueError(
                'eta_a should be less than eta_w for existence of positive steps that satisfy strong Wolfe conditions'
            )

        # max_itr = self.options['max_itr']
        max_step = self.options['max_step']
        min_step = self.options['min_step']
        alpha_tol = self.options['alpha_tol']

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

        alpha, new_f_evals, new_g_evals, f2, f1, g2 = line_search(
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
            amin=min_step,
            xtol=alpha_tol)

        slope2 = np.inner(g2, p)

        converged = True
        if alpha == None:
            converged = False

        num_f_evals += new_f_evals
        num_g_evals += new_g_evals

        return alpha, f2, g2, slope2, num_f_evals, num_g_evals, converged
