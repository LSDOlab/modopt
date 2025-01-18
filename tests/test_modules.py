# import pytest
import numpy as np


def test_line_searches():
    from modopt.line_search_algorithms import Minpack2LS, BacktrackingArmijo

    f = lambda x: (x[0]-0.5)**2 + x[1]**2
    g = lambda x: np.array([2*(x[0]-0.5), 2*x[1]])
    x = np.array([0., 0.])
    p = np.array([1., 0.])

    LS = BacktrackingArmijo(f=f, g=g, eta_a=1e-4, gamma_c=0.4, maxiter=25, max_step=1.)
    alpha, f2, nfev, ngev, converged = LS.search(x, p)

    print(alpha, f2, nfev, ngev, converged)
    assert (alpha, nfev, ngev, converged) == (0.4, 3, 1, True)
    assert np.isclose(f2, 0.01, atol=1e-12)

    LS = Minpack2LS(f=f, g=g, eta_a=1e-4, eta_w=0.9, max_step=1., min_step=1e-12, maxiter=10, alpha_tol=1e-14)
    alpha, f2, g2, slope2,  nfev, ngev, converged = LS.search(x, p)

    print(alpha, f2, g2, slope2, nfev, ngev, converged)
    assert (converged, nfev, ngev) == (True, 3, 3)
    assert np.isclose(f2, 0., atol=1e-8)
    assert np.isclose(slope2, 0., atol=1e-4)

# def test_merit_functions():
#     '''
#     Test merit function modules
#     '''

#     # Import Base class to test
#     from modopt import MeritFunction
#     # Import modules to test
#     from modopt.merit_functions import (
#         L1Eq, L2Eq, LInfEq, 
#         LagrangianEq, 
#         ModifiedLagrangianIneq, 
#         AugmentedLagrangianEq, 
#         AugmentedLagrangianIneq)

#     # Import the problem to test with
#     from examples.ex_8bean_function import BeanFunction

#     prob = BeanFunction()
#     tol = 1e-6
#     maxiter = 200
#     optimizer = SteepestDescent(prob, maxiter=maxiter, opt_tol=tol)
#     optimizer.solve()

#     # Check to make sure values are correct
#     np.testing.assert_almost_equal(
#         actual_val, 
#         desired_val,
#         decimal = 7,
#     )

# def test_hessian_approximations():
#     '''
#     Test Hessian approximation modules
#     '''

#     # Import Base class to test
#     from modopt import ApproximateHessian
#     # Import modules to test
#     from modopt.approximate_hessians import (
#         Broyden, BroydenFirst, BroydenClass, 
#         BFGS, BFGSM1, BFGSScipy, 
#         SR1, DFP, PSB)

#     # Import the problem to test with
#     from examples.ex_8bean_function import BeanFunction

#     prob = BeanFunction()
#     tol = 1e-6
#     maxiter = 200
#     optimizer = SteepestDescent(prob, maxiter=maxiter, opt_tol=tol)
#     optimizer.solve()

#     # Check to make sure values are correct
#     np.testing.assert_almost_equal(
#         actual_val, 
#         desired_val,
#         decimal = 7,
#     )

# def test_trust_region_algorithms():
#     '''
#     Test trust-region modules
#     '''

#     # Import Base class to test
#     from modopt import TrustRegion
#     # Import modules to test
#     from modopt.trust_region_algorithms import SteepestDescentTrustRegion

#     # Import the problem to test with
#     from examples.ex_8bean_function import BeanFunction

#     prob = BeanFunction()
#     tol = 1e-6
#     maxiter = 200
#     optimizer = SteepestDescent(prob, maxiter=maxiter, opt_tol=tol)
#     optimizer.solve()

#     # Check to make sure values are correct
#     np.testing.assert_almost_equal(
#         actual_val, 
#         desired_val,
#         decimal = 7,
#     )

if __name__ == '__main__':
    test_line_searches()
    print('All tests passed!')