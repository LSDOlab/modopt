# Test SLSQP, PySLSQP, SNOPT, IPOPT, COBYLA, and BFGS
# The same tests are used for the sime optimize() function

from all_problem_types import Scaling, scaling_lite, Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

@pytest.mark.slsqp
@pytest.mark.interfaces
def test_slsqp():
    import numpy as np
    from modopt import SLSQP

    prob = Scaling()

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    optimizer = SLSQP(prob, solver_options={'maxiter':50, 'disp':True})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True, optimal_gradient=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_lite_summary.out'})
    optimizer = SLSQP(prob, solver_options={'maxiter':50, 'disp':True}, outputs=['x'])
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True, optimal_gradient=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)

@pytest.mark.cobyla
@pytest.mark.interfaces
def test_cobyla():
    import numpy as np
    from modopt import COBYLA
    from all_problem_types import IneqConstrained, ineq_constrained_lite

    prob = IneqConstrained()
    prob.x0 = np.array([50., 5.])

    optimizer = COBYLA(prob, solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6})
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(optimizer.results['x'], [0.5, -0.5], decimal=6)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=11)
    

    prob = ineq_constrained_lite()
    prob.x0 = np.array([50., 5.])

    optimizer = COBYLA(prob, solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6}, outputs=['x'])
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(optimizer.results['x'], [0.5, -0.5], decimal=6)
    assert_almost_equal(optimizer.results['fun'], 0.125, decimal=6)

    prob = Scaling()

    with pytest.raises(Exception) as exc_info:
        optimizer = COBYLA(prob, solver_options={'maxiter':1000, 'disp':False, 'tol':1e-6})
    
    assert exc_info.type is RuntimeError
    assert str(exc_info.value) == 'Detected equality constraints in the problem. '\
                                  'COBYLA does not support equality constraints. '\
                                  'Use a different solver (PySLSQP, IPOPT, etc.) or remove the equality constraints.'

@pytest.mark.bfgs
@pytest.mark.interfaces
def test_bfgs():
    from modopt import BFGS
    from all_problem_types import BoundConstrained, EqConstrained, IneqConstrained

    prob = BoundConstrained()
    with pytest.raises(Exception) as exc_info:
        optimizer = BFGS(prob, solver_options={'maxiter':200, 'disp':False, 'gtol':1e-6})
    
    assert exc_info.type is RuntimeError
    assert str(exc_info.value) == 'BFGS does not support bounds on variables. ' \
                                  'Please use a different optimizer.'
    
    probs = [EqConstrained(), IneqConstrained()]
    for prob in probs:
        with pytest.raises(Exception) as exc_info:
            optimizer = BFGS(prob, solver_options={'maxiter':200, 'disp':False, 'gtol':1e-6})

        assert exc_info.type is RuntimeError
        assert str(exc_info.value) == 'BFGS does not support constraints. ' \
                                      'Please use a different optimizer.'
    
    from all_problem_types import Unconstrained, unconstrained_lite

    prob = Unconstrained()

    optimizer = BFGS(prob, solver_options={'maxiter':200, 'disp':False, 'gtol':1e-12})
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True, optimal_gradient=True, optimal_hessian_inverse=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(optimizer.results['x'], [0.0, 0.0], decimal=4)
    assert_almost_equal(optimizer.results['fun'], 0.0, decimal=11)
    

    prob = unconstrained_lite()

    optimizer = BFGS(prob, solver_options={'maxiter':200, 'disp':True, 'gtol':1e-12}, outputs=['x', 'obj'])
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(optimal_variables=True, optimal_gradient=True, optimal_hessian_inverse=True)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully.'
    assert_array_almost_equal(optimizer.results['x'], [0.0, 0.0], decimal=4)
    assert_almost_equal(optimizer.results['fun'], 0.0, decimal=11)

@pytest.mark.pyslsqp
@pytest.mark.interfaces
def test_pyslsqp():
    import numpy as np
    from modopt import PySLSQP

    prob = Scaling()

    optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # optimizer = SLSQP(prob, solver_options={'maxiter':50, 'disp':True})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    # assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    assert_almost_equal(optimizer.results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_lite_summary.out'})
    # optimizer = SLSQP(prob, solver_options={'maxiter':50, 'disp':True})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    # assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    assert_almost_equal(optimizer.results['objective'], 20., decimal=6)

@pytest.mark.snopt
@pytest.mark.interfaces
def test_snopt():
    import numpy as np
    from modopt import SNOPT

    prob = Scaling()

    snopt_options = {
        'Infinite bound': 1.0e20,
        'Verify level': 3,
        'Verbose': False,
        'Major optimality': 1e-8
        }
    optimizer = SNOPT(prob, solver_options=snopt_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results.info == 1
    assert_array_almost_equal(optimizer.results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(optimizer.results.objective, 20., decimal=11)
    

    prob = scaling_lite()
    snopt_options.update({'Verbose': True})
    optimizer = SNOPT(prob, solver_options=snopt_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    # print(optimizer.results)
    assert optimizer.results.info == 1
    assert_array_almost_equal(optimizer.results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(optimizer.results.objective, 20., decimal=11)

@pytest.mark.ipopt
@pytest.mark.interfaces
def test_ipopt():
    import numpy as np
    from modopt import IPOPT

    prob = Scaling()
    solver_options = {
        'print_level': 5, 
        'print_frequency_iter': 1, 
        'print_frequency_time': 0, 
        'print_timing_statistics': 'yes'
    }
    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=9)
    assert_almost_equal(optimizer.results['f'], 20., decimal=7)
    assert_almost_equal(optimizer.results['c'], [5, 0.5], decimal=9)
    assert_almost_equal(optimizer.results['lam_c'], [ -5.33333329, -53.33333291], decimal=9)
    assert_almost_equal(optimizer.results['lam_x'], [-2.50165336e-07,  0.], decimal=11)
    

    prob = scaling_lite()

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=9)
    assert_almost_equal(optimizer.results['f'], 20., decimal=7)
    assert_almost_equal(optimizer.results['c'], [5, 0.5], decimal=9)
    assert_almost_equal(optimizer.results['lam_c'], [ -5.33333329, -53.33333291], decimal=9)
    assert_almost_equal(optimizer.results['lam_x'], [-2.50165336e-07,  0.], decimal=11)

    # test unconstrained problem
    # IPOPT performs poorly on the following unconstrained problem. 
    # Need to increase the tolerance to 1e-10 for a decent solution accurate upto 1 decimal.
    prob = Unconstrained()
    solver_options['tol'] = 1e-10

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)

    prob = unconstrained_lite()

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)

@pytest.mark.ipopt
@pytest.mark.interfaces
def test_ipopt_exact_hess_lag():
    from all_problem_types import SecondOrderScaling, second_order_scaling_lite, SecondOrderUnconstrained, second_order_unconstrained_lite
    import numpy as np
    from modopt import IPOPT

    prob = SecondOrderScaling()
    solver_options = {
        'print_level': 5, 
        'print_frequency_iter': 1, 
        'print_frequency_time': 0, 
        'print_timing_statistics': 'yes',
        'hessian_approximation': 'exact',
    }
    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=6)
    assert_almost_equal(optimizer.results['f'], 20., decimal=5)
    assert_almost_equal(optimizer.results['c'], [5, 0.5], decimal=6)
    assert_almost_equal(optimizer.results['lam_c'], [ -5.333334058, -53.333340585], decimal=9)
    assert_almost_equal(optimizer.results['lam_x'], [-4.54553843e-06,  0.], decimal=11)
    

    prob = second_order_scaling_lite()

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=6)
    assert_almost_equal(optimizer.results['f'], 20., decimal=5)
    assert_almost_equal(optimizer.results['c'], [5, 0.5], decimal=6)
    assert_almost_equal(optimizer.results['lam_c'], [ -5.333334058, -53.333340585], decimal=9)
    assert_almost_equal(optimizer.results['lam_x'], [-4.54553843e-06,  0.], decimal=11)

    # test unconstrained problem
    # IPOPT performs poorly on the following unconstrained problem. 
    # Need to increase the tolerance to 1e-10 for a decent solution accurate upto 1 decimal.
    prob = SecondOrderUnconstrained()
    solver_options['tol'] = 1e-10

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)

    prob = second_order_unconstrained_lite()

    optimizer = IPOPT(prob, solver_options=solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)

if __name__ == '__main__':
    test_slsqp()
    test_cobyla()
    test_bfgs()
    test_pyslsqp()
    test_snopt()
    test_ipopt()
    test_ipopt_exact_hess_lag()
    print('All tests passed!')