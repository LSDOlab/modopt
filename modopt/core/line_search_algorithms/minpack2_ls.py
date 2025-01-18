"""
Some of the code in this file is adapted from SciPy
to add more flexibility to the line search algorithm.
The original code is licensed under the BSD 3-Clause "New" or "Revised" License,
and the license can be found at:
https://github.com/scipy/scipy/blob/main/LICENSE.txt .
"""


import numpy as np

eps = np.finfo(np.float64).resolution  # 1.4901e-15

from modopt import LineSearch
# from scipy.optimize._linesearch import line_search_wolfe1 as line_search

# def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
#                        old_fval=None, old_old_fval=None,
#                        args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
#                        xtol=1e-14):

## The following code is copied from scipy.optimize._linesearch
## and modified to include 'maxiter' option in line_search_wolfe1,
## since SciPy does not allow setting 'maxiter' for line_search_wolfe1
## and hardcodes it to 100.

from scipy.optimize._dcsrch import DCSRCH

def _check_c1_c2(c1, c2):
    if not (0 < c1 < c2 < 1):
        raise ValueError("'c1' and 'c2' do not satisfy"
                         "'0 < c1 < c2 < 1'.")


#------------------------------------------------------------------------------
# Minpack's Wolfe line and scalar searches
#------------------------------------------------------------------------------

## Change from SciPy: 'maxiter' option is not available in SciPy
def line_search_wolfe1(f, fprime, xk, pk, gfk=None,
                       old_fval=None, old_old_fval=None,
                       args=(), c1=1e-4, c2=0.9, amax=50, amin=1e-8,
                       xtol=1e-14, maxiter=10):
    """
    As `scalar_search_wolfe1` but do a line search to direction `pk`

    Parameters
    ----------
    f : callable
        Function `f(x)`
    fprime : callable
        Gradient of `f`
    xk : array_like
        Current point
    pk : array_like
        Search direction
    gfk : array_like, optional
        Gradient of `f` at point `xk`
    old_fval : float, optional
        Value of `f` at point `xk`
    old_old_fval : float, optional
        Value of `f` at point preceding `xk`
    maxiter : int, optional
        Maximum number of iterations to perform

    The rest of the parameters are the same as for `scalar_search_wolfe1`.

    Returns
    -------
    stp, f_count, g_count, fval, old_fval
        As in `line_search_wolfe1`
    gval : array
        Gradient of `f` at the final point

    Notes
    -----
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1``.

    """
    if gfk is None:
        gfk = fprime(xk, *args)

    gval = [gfk]
    gc = [0]
    fc = [0]

    def phi(s):
        fc[0] += 1
        return f(xk + s*pk, *args)

    def derphi(s):
        gval[0] = fprime(xk + s*pk, *args)
        gc[0] += 1
        return np.dot(gval[0], pk)

    derphi0 = np.dot(gfk, pk)

    stp, fval, old_fval = scalar_search_wolfe1(
            phi, derphi, old_fval, old_old_fval, derphi0,
            c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol, 
            maxiter=maxiter)

    return stp, fc[0], gc[0], fval, old_fval, gval[0]

## Change from SciPy: 'maxiter' option is not available in SciPy
def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, 
                         derphi0=None, c1=1e-4, c2=0.9, 
                         amax=50, amin=1e-8, xtol=1e-14, maxiter=10):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions

    alpha > 0 is assumed to be a descent direction.

    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.
    maxiter : int, optional
        Maximum number of iterations to perform

    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`

    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    
    Parameters `c1` and `c2` must satisfy ``0 < c1 < c2 < 1`` as described in [1]_.

    References
    ----------
    
    .. [1] Nocedal, J., & Wright, S. J. (2006). Numerical optimization.
       In Springer Series in Operations Research and Financial Engineering.
       (Springer Series in Operations Research and Financial Engineering).
       Springer Nature.

    """
    _check_c1_c2(c1, c2)

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    ## Change from SciPy: maxiter is set as 100 in SciPy 
    # maxiter = 100

    dcsrch = DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)
    stp, phi1, phi0, task = dcsrch(
        alpha1, phi0=phi0, derphi0=derphi0, maxiter=maxiter
    )

    return stp, phi1, phi0

line_search = line_search_wolfe1

class Minpack2LS(LineSearch):
    """
    The Minpack2 line search algorithm for steps that satisfy the strong Wolfe conditions.

    Parameters
    ----------
    f : callable
        Merit function.
    g : callable
        Gradient of the merit function.
    eta_a : float, default=1e-4
        Armijo parameter.
    eta_w : float, default=0.9
        Wolfe parameter.
    max_step : float, default=1.
        Maximum step length.
    min_step : float, default=1e-12
        Minimum step length.
    maxiter : int, default=10
        Maximum number of line search iterations.
    alpha_tol : float, default=1e-14
        Relative tolerance for an acceptable step.
    """
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
        # Minimum step length
        self.options.declare('min_step',
                             default=1e-12,
                             types=float,
                             lower=eps,
                             upper=(1.0 - eps))
        
        # Maximum number of iterations allowed before convergence
        self.options.declare('maxiter',
                             default=10,
                             types=int,
                             lower=1,
                             upper=100)

        # Relative tolerance for an acceptable step
        self.options.declare('alpha_tol',
                             default=1e-14,
                             types=float,
                             lower=eps,
                             upper=(1.0 - eps))

    def search(self, x, p, f0=None, g0=None):
        """
        Perform a line search to find a step length that satisfies the strong Wolfe conditions.

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

        Returns
        -------
        alpha : float
            Step length found by the line search.
        f2 : float
            Value of the merit function at the new point.
        g2 : np.ndarray
            Gradient of the merit function at the new point.
        slope2 : float
            Slope of the merit function at the new point along the search direction.
        nfev : int
            Number of additional function evaluations.
        ngev : int
            Number of additional gradient evaluations.
        converged : bool
            ``True`` if the line search converged to a step length
            that satisfies the strong Wolfe conditions, ``False`` otherwise.
        """

        eta_a = self.options['eta_a']
        eta_w = self.options['eta_w']

        if eta_a > eta_w:
            raise ValueError(
                'eta_a should be less than eta_w for existence of positive steps that satisfy strong Wolfe conditions'
            )

        maxiter   = self.options['maxiter']
        max_step  = self.options['max_step']
        min_step  = self.options['min_step']
        alpha_tol = self.options['alpha_tol']

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
            xtol=alpha_tol,
            maxiter=maxiter)

        slope2 = np.inner(g2, p)

        converged = True
        if alpha == None:
            converged = False

        nfev += new_f_evals
        ngev += new_g_evals

        return alpha, f2, g2, slope2, nfev, ngev, converged
