import pytest
import numpy as np


def test_unconstrained():
    '''
    Test gradient-based algorithms for unconstrained optimization
    '''

    # Import algorithms to test
    from modopt import SteepestDescent, Newton, QuasiNewton

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

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = Newton(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = QuasiNewton(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )


def test_grad_free_continuous():
    '''
    Test gradient free optimization algorithms for continuous variables
    '''

    # Import algorithms to test
    from modopt import SteepestDescent, Newton, QuasiNewton

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


def test_grad_free_continuous():
    '''
    Test gradient free optimization algorithms for continuous variables
    '''

    # Import algorithms to test
    from modopt import PSO, NelderMead

    # Import the problem to test with
    from examples.ex_8bean_function import BeanFunction

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = PSO(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

    prob = BeanFunction()
    tol = 1e-6
    max_itr = 200
    optimizer = NelderMead(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )


def test_grad_free_discrete():
    '''
    Test gradient free optimization algorithms for continuous variables
    '''

    # Import algorithms to test
    from modopt import SimulatedAnnealing

    # Import the problem to test with
    from examples.ex_9traveling_salesman import TravelingSalesman

    prob = TravelingSalesman()
    tol = 1e-6
    max_itr = 200
    optimizer = SimulatedAnnealing(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )


def test_eq_constrained():
    '''
    Test gradient-based algorithms for 
    equality-constrained optimization
    '''

    # Import algorithms to test
    from modopt import NewtonLagrange, L2PenaltyEq

    # Import the problem to test with
    from examples.ex_9traveling_salesman import TravelingSalesman

    prob = TravelingSalesman()
    tol = 1e-6
    max_itr = 200
    optimizer = NewtonLagrange(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )


def test_ineq_constrained():
    '''
    Test gradient-based algorithms for general,
    inequality-constrained optimization
    '''

    # Import algorithms to test
    from modopt import SQP

    # Import the problem to test with
    from examples.ex_9traveling_salesman import TravelingSalesman

    prob = TravelingSalesman()
    tol = 1e-6
    max_itr = 200
    optimizer = SQP(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

def test_scipy_algorithms():
    '''
    Test optimization algorithms interfaced 
    from scipy.optimize
    '''

    # Import algorithms to test
    from modopt import SLSQP, COBYLA

    # Import the problem to test with
    from examples.ex_9traveling_salesman import TravelingSalesman

    prob = TravelingSalesman()
    tol = 1e-6
    max_itr = 200
    optimizer = SLSQP(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

def test_snopt():
    '''
    Test SNOPT-C interface
    '''

    # Import algorithms to test
    from modopt import SNOPT

    # Import the problem to test with
    from examples.ex_9traveling_salesman import TravelingSalesman

    prob = TravelingSalesman()
    tol = 1e-6
    max_itr = 200
    optimizer = SNOPT(prob, max_itr=max_itr, opt_tol=tol)
    optimizer.solve()

    # Check to make sure values are correct
    np.testing.assert_almost_equal(
        actual_val, 
        desired_val,
        decimal = 7,
    )

# def test_slsqp_v2():
#     '''
#     Test the built-in revamped version of SLSQP
#     '''

#     # Import algorithms to test
#     from modopt import SLSQPv2

#     # Import the problem to test with
#     from examples.ex_9traveling_saleman import TravelingSalesman

#     prob = TravelingSalesman()
#     tol = 1e-6
#     max_itr = 200
#     optimizer = SLSQPv2(prob, max_itr=max_itr, opt_tol=tol)
#     optimizer.solve()

#     # Check to make sure values are correct
#     np.testing.assert_almost_equal(
#         actual_val, 
#         desired_val,
#         decimal = 7,
#     )