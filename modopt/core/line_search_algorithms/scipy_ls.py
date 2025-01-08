import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt import LineSearch
from scipy.optimize import line_search

# This line search from SciPy is buggy.
# To see the bug, run `test_sqp.py` for the Constrained or constrained_lite problems.
# _zoom function evaluates the objective function at the same point 10+ times
# whenever it fails and then prints a warning message saying failed to converge.

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
                             types=float,
                             lower=eps,
                             upper=50.)

        # Maximum number of iterations allowed before convergence
        self.options.declare('maxiter',
                             default=8,
                             types=int,
                             lower=1,
                             upper=100)

    def search(self, x, p, f0=None, g0=None):

        eta_a = self.options['eta_a']
        eta_w = self.options['eta_w']

        if eta_a > eta_w:
            raise ValueError(
                'eta_a should be less than eta_w for existence of positive steps that satisfy strong Wolfe conditions'
            )

        maxiter  = self.options['maxiter']
        max_step = self.options['max_step']

        f = self.options['f']
        g = self.options['g']

        nfev = 0
        ngev = 0

        if f0 is None:
            f1 = f(x)
            nfev = 1
        else:
            f1 = f0 * 1.

        if g0 is None:
            g1 = g(x)
            ngev = 1
        else:
            g1 = g0 * 1.

        alpha, new_f_evals, new_g_evals, f2, f1, slope2 = line_search(
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
            maxiter=maxiter)

        g2 = "Unavailable"

        converged = True
        if alpha is None:
            converged = False

        nfev += new_f_evals
        ngev += new_g_evals

        return alpha, f2, g2, slope2, nfev, ngev, converged
