# Test SLSQP, PySLSQP, SNOPT, COBYLA, and BFGS

from all_problem_types import Scaling, scaling_lite
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


if __name__ == '__main__':
    test_slsqp()
    test_pyslsqp()
    test_snopt()
    print('All tests passed!')