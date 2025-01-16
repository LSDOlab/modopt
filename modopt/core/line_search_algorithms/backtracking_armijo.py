import numpy as np

eps = np.finfo(np.float64).resolution  # 1e-15

from modopt import LineSearch


class BacktrackingArmijo(LineSearch):
    """
    Backtracking line search for steps that satisy the Armijo condition.
    
    Parameters
    ----------
    f : callable
        Merit function.
    g : callable
        Gradient of the merit function.
    eta_a : float, default=1e-4
        Armijo parameter.
    gamma_c : float, default=0.3
        Step length contraction factor.
    maxiter : int, default=25
        Maximum number of line search iterations.
    max_step : float, default=1.
        Maximum step length.
    """
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
        """
        Perform a backtracking line search to find a step length that satisfies the Armijo condition.

        Parameters
        ----------
        x : np.ndarray
            Current point.
        p : np.ndarray
            Search direction.
        f0 : float, optional
            Value of the merit function at the current point.
        g0 : np.ndarray, optional
            Gradient of the merit function at the current point.
        """

        eta_a    = self.options['eta_a']
        gamma_c  = self.options['gamma_c']
        maxiter  = self.options['maxiter']
        max_step = self.options['max_step']
        f = self.options['f']
        g = self.options['g']

        alpha = max_step * 1.
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

        return alpha, f2, nfev, ngev, converged