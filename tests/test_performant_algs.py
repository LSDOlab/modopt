# Test SLSQP, PySLSQP, SNOPT, COBYLA, and BFGS

from all_problem_types import Scaling, scaling_lite, Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

def test_slsqp():
    import numpy as np
    from modopt import SLSQP

    prob = Scaling()

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)
    

    prob = scaling_lite()

    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_lite_summary.out'})
    optimizer = SLSQP(prob, maxiter=50, disp=True)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [2., 0.], decimal=8)
    assert_almost_equal(optimizer.results['fun'], 20., decimal=6)
    # assert_almost_equal(optimizer.results['objective'], 20., decimal=6)

@pytest.mark.pyslsqp
@pytest.mark.interfaces
def test_pyslsqp():
    import numpy as np
    from modopt import PySLSQP

    prob = Scaling()

    optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20, 'summary_filename':'scaling_summary.out'})
    # optimizer = SLSQP(prob, maxiter=50, disp=True)
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
    # optimizer = SLSQP(prob, maxiter=50, disp=True)
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

    optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=False, Major_optimality=1e-8)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    assert optimizer.results.info == 1
    assert_array_almost_equal(optimizer.results.x[:prob.nx], [2., 0.], decimal=11)
    assert_almost_equal(optimizer.results.objective, 20., decimal=11)
    

    prob = scaling_lite()

    optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True, Major_optimality=1e-8)
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
    test_pyslsqp()
    test_snopt()
    test_ipopt()
    test_ipopt_exact_hess_lag()
    print('All tests passed!')