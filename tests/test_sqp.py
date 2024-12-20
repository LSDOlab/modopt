# Test the SQP optimizer

from all_problem_types import Constrained, constrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

@pytest.mark.sqp
def test_sqp(): 
    import numpy as np
    from modopt import SQP

    prob = Constrained()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-8, 'feas_tol': 1e-8}
    optimizer = SQP(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=10)
    assert_array_almost_equal(optimizer.results['c'], [1., 0., 0., 0.], decimal=11)
    assert_array_almost_equal(optimizer.results['pi'], [0., 1.7836, 1.3333, 0.4503], decimal=3)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=9)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 1351
    # assert optimizer.results['nfev'] == 394
    assert optimizer.results['ngev'] == 271
    
    prob = constrained_lite()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = SQP(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=10)
    assert_array_almost_equal(optimizer.results['c'], [1., 0., 0., 0.], decimal=11)
    assert_array_almost_equal(optimizer.results['pi'], [0., 1.7836, 1.3333, 0.4503], decimal=3)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=9)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 1351
    # assert optimizer.results['nfev'] == 394
    assert optimizer.results['ngev'] == 271

if __name__ == '__main__':
    test_sqp()
    print('All tests passed!')