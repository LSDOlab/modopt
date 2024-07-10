# Test the optimize function interface for SLSQP, PySLSQP, SNOPT, IPOPT, CVXOPT,
# ConvexQPSolvers, COBYLA, and BFGS
# The tests are exactly the same as in test_performant_algs.py, test_qpsolvers.py, and test_cvxopt.py

from all_problem_types import Scaling, scaling_lite, Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest
from modopt import optimize

@pytest.mark.slsqp
@pytest.mark.interfaces
def test_slsqp():

    prob = Scaling()

    # results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    results = optimize(prob, solver='SLSQP', solver_options={'maxiter':50, 'disp':True})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    assert_almost_equal(results['fun'], 20., decimal=6)
    # assert_almost_equal(results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    # results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    results = optimize(prob, solver='SLSQP', solver_options={'maxiter':50, 'disp':True})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    assert_almost_equal(results['fun'], 20., decimal=6)
    # assert_almost_equal(results['objective'], 20., decimal=6)

@pytest.mark.cobyla
@pytest.mark.interfaces
def test_cobyla():
    import numpy as np
    from modopt import COBYLA
    from all_problem_types import IneqConstrained, ineq_constrained_lite

    prob = IneqConstrained()
    prob.x0 = np.array([50., 5.])

    results = optimize(prob, solver='COBYLA', solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6})
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [0.5, -0.5], decimal=6)
    assert_almost_equal(results['fun'], 0.125, decimal=11)
    

    prob = ineq_constrained_lite()
    prob.x0 = np.array([50., 5.])

    results = optimize(prob, solver='COBYLA', solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6}, outputs=['x'])
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [0.5, -0.5], decimal=6)
    assert_almost_equal(results['fun'], 0.125, decimal=6)

@pytest.mark.bfgs
@pytest.mark.interfaces
def test_bfgs():
    from all_problem_types import Unconstrained, unconstrained_lite

    prob = Unconstrained()

    results = optimize(prob, solver="BFGS", solver_options={'maxiter':200, 'disp':False, 'gtol':1e-12})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [0.0, 0.0], decimal=4)
    assert_almost_equal(results['fun'], 0.0, decimal=11)

    prob = unconstrained_lite()

    results = optimize(prob, solver="BFGS", solver_options={'maxiter':200, 'disp':True, 'gtol':1e-12}, outputs=['x', 'obj'])
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [0.0, 0.0], decimal=4)
    assert_almost_equal(results['fun'], 0.0, decimal=11)

@pytest.mark.lbfgsb
@pytest.mark.interfaces
def test_lbfgsb():
    import numpy as np
    from all_problem_types import BoundConstrained, bound_constrained_lite

    prob = BoundConstrained()

    results = optimize(prob, solver='LBFGSB', solver_options={'maxiter':200, 'iprint':-1, 'gtol':1e-8, 'ftol':1e-12})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
    assert_array_almost_equal(results['x'], [1.0, 0.0], decimal=3)
    assert_almost_equal(results['fun'], 1.0, decimal=11)
    

    prob = bound_constrained_lite()

    optimizer = optimize(prob, solver='LBFGSB', solver_options={'maxiter':200, 'iprint':1, 'gtol':1e-8, 'ftol':1e-12}, outputs=['x', 'obj'])
    print(results)
    assert results['success'] == True
    assert results['message'] == 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
    assert_array_almost_equal(results['x'], [1.0, 0.0], decimal=3)
    assert_almost_equal(results['fun'], 1.0, decimal=11)

@pytest.mark.pyslsqp
@pytest.mark.interfaces
def test_pyslsqp():

    prob = Scaling()

    results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # results = optimize(prob, solver='SLSQP', solver_options={'maxiter':50, 'disp':True})
    print(results)
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    # assert_almost_equal(results['fun'], 20., decimal=6)
    assert_almost_equal(results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # results = optimize(prob, solver='SLSQP', solver_options={'maxiter':50, 'disp':True})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    # assert_almost_equal(results['fun'], 20., decimal=6)
    assert_almost_equal(results['objective'], 20., decimal=6)

@pytest.mark.snopt
@pytest.mark.interfaces
def test_snopt():

    prob = Scaling()

    snopt_options = {
        'Infinite bound': 1.0e20, 
        'Verify level': 3,
        'Verbose': False,
        'Major optimality': 1e-8
    }

    results = optimize(prob, solver='SNOPT', solver_options=snopt_options)
    print(results)
    assert results.info == 1
    assert_array_almost_equal(results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(results.objective, 20., decimal=11)
    

    prob = scaling_lite()

    results = optimize(prob, solver='SNOPT', solver_options=snopt_options)
    # print(results)
    assert results.info == 1
    assert_array_almost_equal(results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(results.objective, 20., decimal=11)

@pytest.mark.ipopt
@pytest.mark.interfaces
def test_ipopt():

    prob = Scaling()
    solver_options = {
        'print_level': 5, 
        'print_frequency_iter': 1, 
        'print_frequency_time': 0, 
        'print_timing_statistics': 'yes'
    }
    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [2., 0.], decimal=9)
    assert_almost_equal(results['f'], 20., decimal=7)
    assert_almost_equal(results['c'], [5, 0.5], decimal=9)
    assert_almost_equal(results['lam_c'], [ -5.33333329, -53.33333291], decimal=9)
    assert_almost_equal(results['lam_x'], [-2.50165336e-07,  0.], decimal=11)
    

    prob = scaling_lite()

    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [2., 0.], decimal=9)
    assert_almost_equal(results['f'], 20., decimal=7)
    assert_almost_equal(results['c'], [5, 0.5], decimal=9)
    assert_almost_equal(results['lam_c'], [ -5.33333329, -53.33333291], decimal=9)
    assert_almost_equal(results['lam_x'], [-2.50165336e-07,  0.], decimal=11)

    # test unconstrained problem
    # IPOPT performs poorly on the following unconstrained problem. 
    # Need to increase the tolerance to 1e-10 for a decent solution accurate upto 1 decimal.
    prob = Unconstrained()
    solver_options['tol'] = 1e-10

    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=1)

    prob = unconstrained_lite()

    print(results)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=1)

@pytest.mark.ipopt
@pytest.mark.interfaces
def test_ipopt_exact_hess_lag():
    from all_problem_types import SecondOrderScaling, second_order_scaling_lite, SecondOrderUnconstrained, second_order_unconstrained_lite

    prob = SecondOrderScaling()
    solver_options = {
        'print_level': 5, 
        'print_frequency_iter': 1, 
        'print_frequency_time': 0, 
        'print_timing_statistics': 'yes',
        'hessian_approximation': 'exact',
    }
    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [2., 0.], decimal=6)
    assert_almost_equal(results['f'], 20., decimal=5)
    assert_almost_equal(results['c'], [5, 0.5], decimal=6)
    assert_almost_equal(results['lam_c'], [ -5.333334058, -53.333340585], decimal=9)
    assert_almost_equal(results['lam_x'], [-4.54553843e-06,  0.], decimal=11)
    

    prob = second_order_scaling_lite()

    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [2., 0.], decimal=6)
    assert_almost_equal(results['f'], 20., decimal=5)
    assert_almost_equal(results['c'], [5, 0.5], decimal=6)
    assert_almost_equal(results['lam_c'], [ -5.333334058, -53.333340585], decimal=9)
    assert_almost_equal(results['lam_x'], [-4.54553843e-06,  0.], decimal=11)

    # test unconstrained problem
    # IPOPT performs poorly on the following unconstrained problem. 
    # Need to increase the tolerance to 1e-10 for a decent solution accurate upto 1 decimal.
    prob = SecondOrderUnconstrained()
    solver_options['tol'] = 1e-10

    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=1)

    prob = second_order_unconstrained_lite()

    results = optimize(prob, solver='IPOPT', solver_options=solver_options)
    print(results)
    assert_array_almost_equal(results['x'], [0., 0.], decimal=1)

@pytest.mark.interfaces
@pytest.mark.qpsolvers
def test_qpsolvers(): 
    from all_problem_types import ConvexQP, convex_qp_lite

    probs = [ConvexQP(), convex_qp_lite()]
    solver_options = {'solver':'quadprog', 'verbose':True}

    for prob in probs:
        results = optimize(prob, solver='ConvexQPSolvers', solver_options=solver_options)
        print(results)

        assert results['found']
        assert_array_almost_equal(results['x'], [1., 0.], decimal=11)
        assert_array_almost_equal(results['z_box'], [0., 0.], decimal=11)
        assert_almost_equal(results['objective'], 1., decimal=11)
        assert_almost_equal(results['primal_residual'], 4.44e-16, decimal=11)
        assert_almost_equal(results['dual_residual'], 0., decimal=11)
        assert_almost_equal(results['duality_gap'], 4.44e-16, decimal=11)
        assert_array_almost_equal(results['constraints'], [1., 1.], decimal=11)
        assert_array_almost_equal(results['y'], [-1.], decimal=11)    # dual variables for the equality constraints
        assert_array_almost_equal(results['z'], [1.], decimal=11)     # dual variables for the inequality constraints
        assert_array_almost_equal(results['extras']['iact'], [1, 2])
        assert_array_almost_equal(results['extras']['iterations'], [3, 0])

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_cvxopt(): 
    from test_cvxopt import ConstrainedBoundedConvex, constrained_bounded_convex_lite

    probs = [ConstrainedBoundedConvex(), constrained_bounded_convex_lite()]
    solver_options = {'maxiters':50, 'abstol':1e-12, 'reltol':1e-12, 'feastol':1e-12}

    for prob in probs:
        results = optimize(prob, solver='CVXOPT', solver_options=solver_options)
        print(results)

        assert results['status'] == 'optimal'
        assert_array_almost_equal(results['x'], [0.5, 0.5], decimal=11)
        assert_almost_equal(results['objective'], 0.5, decimal=11)
        assert_array_almost_equal(results['constraints'], [1., -0.25], decimal=11)

def test_invalid_solver():
    prob = Scaling()
    with pytest.raises(Exception) as exc_info:
        optimize(prob, solver='InvalidSolver')

    assert exc_info.type is ValueError
    assert str(exc_info.value) == "Invalid solver named 'InvalidSolver' is specified. Valid solvers are: "\
                                  "['SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 'LBFGSB', "\
                                  "'SNOPT', 'IPOPT', 'CVXOPT', 'ConvexQPSolvers']."

if __name__ == '__main__':
    test_slsqp()
    test_cobyla()
    test_bfgs()
    test_lbfgsb()
    test_pyslsqp()
    test_snopt()
    test_ipopt()
    test_ipopt_exact_hess_lag()
    test_qpsolvers()
    test_cvxopt()
    test_invalid_solver()
    print('All tests passed!')