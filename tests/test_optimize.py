# Test the optimize function interface for SLSQP, PySLSQP, SNOPT, IPOPT, CVXOPT,
# ConvexQPSolvers, COBYLA, BFGS, LBFGSB, and NelderMead solvers.
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

    results = optimize(prob, solver='COBYLA', solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6}, readable_outputs=['x'])
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

    results = optimize(prob, solver="BFGS", solver_options={'maxiter':200, 'disp':True, 'gtol':1e-12}, readable_outputs=['x', 'obj'])
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

    optimizer = optimize(prob, solver='LBFGSB', solver_options={'maxiter':200, 'iprint':1, 'gtol':1e-8, 'ftol':1e-12}, readable_outputs=['x', 'obj'])
    print(results)
    assert results['success'] == True
    assert results['message'] == 'CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL'
    assert_array_almost_equal(results['x'], [1.0, 0.0], decimal=3)
    assert_almost_equal(results['fun'], 1.0, decimal=11)

@pytest.mark.nelder_mead
@pytest.mark.interfaces
def test_nelder_mead():
    from all_problem_types import BoundConstrained, bound_constrained_lite

    prob = BoundConstrained()

    results = optimize(prob, solver='NelderMead', solver_options={'maxiter':200, 'disp':False, 'fatol':1e-6, 'xatol':1e-6})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [1.0, 0.0], decimal=4)
    assert_almost_equal(results['fun'], 1.0, decimal=11)
    
    prob = bound_constrained_lite()

    results = optimize(prob, solver='NelderMead', solver_options={'maxiter':200, 'disp':True, 'fatol':1e-6, 'xatol':1e-6}, readable_outputs=['x', 'obj'])
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(results['x'], [1.0, 0.0], decimal=4)
    assert_almost_equal(results['fun'], 1.0, decimal=11)

@pytest.mark.cobyqa
@pytest.mark.interfaces
def test_cobyqa():
    prob = Scaling()

    results = optimize(prob, solver='COBYQA', solver_options={'maxiter':1000, 'debug':True, 'disp':False, 'feasibility_tol':1e-8})
    print(results)
    assert results['success'] == True
    assert results['message'] == 'The lower bound for the trust-region radius has been reached'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=9)
    assert_almost_equal(results['fun'], 20., decimal=7)
    
    prob = scaling_lite()

    results = optimize(prob, solver='COBYQA', solver_options={'maxiter':1000, 'disp':True, 'feasibility_tol':1e-8, 'store_history':True}, readable_outputs=['x', 'obj'])
    print(results)
    assert results['success'] == True
    assert results['message'] == 'The lower bound for the trust-region radius has been reached'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=9)
    assert_almost_equal(results['fun'], 20., decimal=7)

@pytest.mark.trust_constr
@pytest.mark.interfaces
def test_trust_constr():
    import numpy as np
    from modopt import TrustConstr

    prob = Scaling()

    results = optimize(prob, solver='TrustConstr', solver_options={'maxiter':1000, 'verbose':0, 'gtol':1e-15})
    print(results)
    assert results['success'] == True
    assert results['message'] == '`xtol` termination condition is satisfied.'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=11)
    assert_almost_equal(results['fun'], 20., decimal=11)
    
    prob = scaling_lite()

    results = optimize(prob, solver='TrustConstr', solver_options={'maxiter':1000, 'verbose':3, 'gtol':1e-15}, readable_outputs=['x', 'obj', 'opt', 'time', 'grad'])
    print(results)
    assert results['success'] == True
    assert results['message'] == '`xtol` termination condition is satisfied.'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=11)
    assert_almost_equal(results['fun'], 20., decimal=11)

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
                                  "['SLSQP', 'PySLSQP', 'COBYLA', 'BFGS', 'LBFGSB', 'NelderMead', 'COBYQA', "\
                                  "'TrustConstr', 'SNOPT', 'IPOPT', 'CVXOPT', 'ConvexQPSolvers']."

if __name__ == '__main__':
    test_slsqp()
    test_cobyla()
    test_bfgs()
    test_lbfgsb()
    test_nelder_mead()
    test_cobyqa()
    test_trust_constr()
    test_pyslsqp()
    test_snopt()
    test_ipopt()
    test_qpsolvers()
    test_cvxopt()
    test_invalid_solver()
    print('All tests passed!')