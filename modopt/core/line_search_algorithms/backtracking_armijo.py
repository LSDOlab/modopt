import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt import LineSearch


class BacktrackingArmijo(LineSearch):
    def initialize(self):
        # Stepsize contraction factor
        self.options.declare('gamma_c',
                             default=0.3,
                             types=float,
                             lower=eps,
                             upper=(1.0 - eps))
        
        # Maximum number of iterations allowed before convergence
        self.options.declare('maxiter',
                             default=25,
                             types=int,
                             lower=1,
                             upper=100)
        
        # Maximum step length
        self.options.declare('max_step',
                             default=1.,
                             types=float,
                             lower=eps,
                             upper=50.)

    def search(self, x, p, f0=None, g0=None):

        eta_a   = self.options['eta_a']
        gamma_c = self.options['gamma_c']
        maxiter = self.options['maxiter']
        f = self.options['f']
        g = self.options['g']

        alpha = self.options['max_step']
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

        slope = np.inner(g1, p)

        itr = 1
        rho = 0.
        while 1:
            f2 = f(x + alpha * p)
            rho = (f2 - f1) / (alpha * slope)

            if (itr <= maxiter) and (rho < eta_a):
                alpha *= gamma_c
                itr += 1
            else:
                break

        nfev += itr

        converged = True
        if rho < eta_a:
            converged = False

        # If converged, try for a better stepsize using
        # arithmetic/geometric mean of (alpha, alpha/gamma_c)
        if converged:
            alpha_am = 0.5 * alpha * (1. + 1./gamma_c)
            f_am     = f(x + alpha_am * p)
            rho_am   = (f_am - f1) / (alpha_am * slope)
            nfev += 1
            if rho_am >= eta_a:
                alpha, f2, rho = alpha_am, f_am, rho_am

            else:
                alpha_gm = alpha / np.sqrt(gamma_c)
                f_gm     = f(x + alpha_gm * p)
                rho_gm   = (f_gm - f1) / (alpha_gm * slope)
                nfev += 1
                if rho_gm >= eta_a:
                    alpha, f2, rho = alpha_gm, f_gm, rho_gm

        return alpha, f2, nfev, ngev, converged