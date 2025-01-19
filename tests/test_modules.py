# import pytest
import numpy as np


def test_line_searches():
    """
    Unit tests for line search algorithms
    """
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

def test_merit_functions():
    """
    Unit tests for merit functions
    """
    from modopt.merit_functions import AugmentedLagrangianIneq

    f = lambda x: (x[0]-0.5)**2 + x[1]**2
    g = lambda x: np.array([2*(x[0]-0.5), 2*x[1]])
    c = lambda x: np.array([x[0] + x[1] - 1.]) # x[0] + x[1] - 1 >= 0
    j = lambda x: np.array([[1., 1.]])

    x = np.array([0., 0.])
    l = np.array([1.])
    s = np.array([0.5])

    merit = AugmentedLagrangianIneq(f=f, g=g, c=c, j=j, nx=2, nc=1)
    assert merit.rho == np.array([0.])
    assert merit.cache == {}
    assert merit.eval_count == {'f': 0, 'g': 0, 'c': 0, 'j': 0}

    merit.set_rho(1.)
    assert merit.rho == np.array([1.])

    # AL = f(x) - l.T @ (c(x)-s) + merit.rho/2 * np.linalg.norm(c(x)-s)**2 
    # 0.25 - 1.*(-1.-0.5) + 1./2 * (-1.-0.5)**2 = 2.875
    AL_f = 2.875
    print('AL_f', AL_f)

    AL_cf = merit.compute_function(np.concatenate((x, l, s)))
    print('AL_cf', AL_cf)
    assert AL_cf == AL_f

    AL_ef = merit.evaluate_function(x, l, s, f(x), c(x))
    print('AL_ef', AL_ef)
    assert AL_ef == AL_f

    # Gradient of AL with respect to x, l and s
    grad_x = g(x) - j(x).T @ l + merit.rho * j(x).T @ (c(x)-s)
    grad_l = -(c(x)-s)
    grad_s = l - merit.rho * (c(x)-s)
    AL_g = np.concatenate((grad_x, grad_l, grad_s))
    print('AL_g', AL_g)

    # Finite difference gradient of AL with respect to x, l and s
    v = np.concatenate((x, l, s))
    h = 1e-8
    AL_g_fd = -merit.compute_function(v) * np.ones_like(v)
    for i in range(len(v)):         # x changes 3 times, so 3 new f/c(x) evaluations
        e = np.zeros_like(v)
        e[i] = h
        AL_g_fd[i] += merit.compute_function(v+e)
        AL_g_fd[i] /= h

    print('AL_g_fd', AL_g_fd)
    assert np.allclose(AL_g_fd, AL_g, atol=1e-6)


    AL_cg = merit.compute_gradient(np.concatenate((x, l, s)))
    print('AL_cg', AL_cg)
    assert np.allclose(AL_cg, AL_g, atol=1e-14)

    AL_eg = merit.evaluate_gradient(x, l, s, f(x), c(x), g(x), j(x))
    print('AL_eg', AL_eg)
    assert np.allclose(AL_eg, AL_g, atol=1e-14)

    assert np.allclose(merit.cache['f'][0], x, atol=1e-14)
    assert np.allclose(merit.cache['g'][0], x, atol=1e-14)
    assert np.allclose(merit.cache['c'][0], x, atol=1e-14)
    assert np.allclose(merit.cache['j'][0], x, atol=1e-14)

    assert merit.cache['f'][1] == 0.25
    assert np.allclose(merit.cache['g'][1], [-1., 0.], atol=1e-14)
    assert np.allclose(merit.cache['c'][1], -1., atol=1e-14)
    assert np.allclose(merit.cache['j'][1], [1., 1.], atol=1e-14)

    assert merit.rho == np.array([1.])
    print('merit.eval_count', merit.eval_count)
    assert merit.eval_count == {'f': 1+2+1, 'g': 1, 'c': 1+2+1, 'j': 1}

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
    test_merit_functions()
    print('All tests passed!')