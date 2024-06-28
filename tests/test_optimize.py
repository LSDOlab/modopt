# Test the optimize function interface for SLSQP, PySLSQP, SNOPT, COBYLA, and BFGS
# The tests are exactly the same as in test_performant_algs.py

from all_problem_types import Scaling, scaling_lite, Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest
from modopt import optimize

def test_slsqp():

    prob = Scaling()

    # results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    results = optimize(prob, solver='SLSQP', maxiter=50, disp=True)
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    assert_almost_equal(results['fun'], 20., decimal=6)
    # assert_almost_equal(results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    # results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    results = optimize(prob, solver='SLSQP', maxiter=50, disp=True)
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    assert_almost_equal(results['fun'], 20., decimal=6)
    # assert_almost_equal(results['objective'], 20., decimal=6)

@pytest.mark.pyslsqp
@pytest.mark.interfaces
def test_pyslsqp():

    prob = Scaling()

    results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # results = optimize(prob, solver='SLSQP', maxiter=50, disp=True)
    print(results)
    print(results)
    assert results['success'] == True
    assert results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(results['x'], [2., 0.], decimal=8)
    # assert_almost_equal(results['fun'], 20., decimal=6)
    assert_almost_equal(results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    results = optimize(prob, solver='PySLSQP', solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # results = optimize(prob, solver='SLSQP', maxiter=50, disp=True)
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

    results = optimize(prob, solver='SNOPT', Infinite_bound=1.0e20, Verify_level=3, Verbose=False, Major_optimality=1e-8)
    print(results)
    assert results.info == 1
    assert_array_almost_equal(results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(results.objective, 20., decimal=11)
    

    prob = scaling_lite()

    results = optimize(prob, solver='SNOPT', Infinite_bound=1.0e20, Verify_level=3, Verbose=False, Major_optimality=1e-8)
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

def test_invalid_solver():
    prob = Scaling()
    with pytest.raises(Exception) as exc_info:
        optimize(prob, solver='InvalidSolver')

    assert exc_info.type is ValueError
    assert str(exc_info.value) == "Invalid solver named 'InvalidSolver' is specified. Valid solvers are: ['SLSQP', 'PySLSQP', 'SNOPT', 'IPOPT']."

if __name__ == '__main__':
    test_slsqp()
    test_pyslsqp()
    test_snopt()
    test_ipopt()
    test_ipopt_exact_hess_lag()
    test_invalid_solver()
    print('All tests passed!')