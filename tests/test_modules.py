import pytest
import numpy as np


def test_line_searches():
    '''
    Test line-search modules
    '''

    # Import Base class to test
    from modopt import LineSearch
    # Import modules to test
    from modopt.line_search_algorithms import ScipyLS, Minpack2LS, BacktrackingWolfe, BacktrackingArmijo

    # Import the problem to test with
    from examples.ex_8bean_function import BeanFunction

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = SteepestDescent(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

def test_merit_functions():
    '''
    Test merit function modules
    '''

    # Import Base class to test
    from modopt import MeritFunction
    # Import modules to test
    from modopt.merit_functions import (
        L1Eq, L2Eq, LInfEq, 
        LagrangianEq, 
        ModifiedLagrangianIneq, 
        AugmentedLagrangianEq, 
        AugmentedLagrangianIneq)

    # Import the problem to test with
    from examples.ex_8bean_function import BeanFunction

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = SteepestDescent(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

def test_hessian_approximations():
    '''
    Test Hessian approximation modules
    '''

    # Import Base class to test
    from modopt import ApproximateHessian
    # Import modules to test
    from modopt.approximate_hessians import (
        Broyden, BroydenFirst, BroydenClass, 
        BFGS, BFGSM1, BFGSScipy, 
        SR1, DFP, PSB)

    # Import the problem to test with
    from examples.ex_8bean_function import BeanFunction

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = SteepestDescent(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

def test_trust_region_algorithms():
    '''
    Test trust-region modules
    '''

    # Import Base class to test
    from modopt import TrustRegion
    # Import modules to test
    from modopt.trust_region_algorithms import SteepestDescentTrustRegion

    # Import the problem to test with
    from examples.ex_8bean_function import BeanFunction

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = SteepestDescent(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

