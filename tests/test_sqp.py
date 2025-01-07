# Test the SQP optimizer

from all_problem_types import Constrained, constrained_lite, Unconstrained, unconstrained_lite
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
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=8)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=7)
    assert_array_almost_equal(optimizer.results['c'], [1., 0., 0., 0.], decimal=7)
    # assert_array_almost_equal(optimizer.results['pi'], [0., 1.851, 1.33333, 0.518], decimal=3)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=8)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=8)
    assert optimizer.results['niter'] < solver_options['maxiter']
    # assert optimizer.results['nfev'] == 73, 140
    # assert optimizer.results['ngev'] == 8, 10
    
    prob = constrained_lite()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = SQP(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=8)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=7)
    assert_array_almost_equal(optimizer.results['c'], [1., 0., 0., 0.], decimal=7)
    # assert_array_almost_equal(optimizer.results['pi'], [0., 1.7532, 1.33333, 0.41987], decimal=3)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=8)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=8)
    assert optimizer.results['niter'] < solver_options['maxiter']
    # assert optimizer.results['nfev'] == 73, 140
    # assert optimizer.results['ngev'] == 8, 10

    prob = Unconstrained()
    prob.x0 = np.array([1., 1.])

    optimizer = SQP(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=6)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=4)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 23
    assert optimizer.results['ngev'] == 21

    prob = unconstrained_lite()
    prob.x0 = np.array([1., 1.])

    optimizer = SQP(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=6)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=4)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 23
    assert optimizer.results['ngev'] == 21

if __name__ == '__main__':
    test_sqp()
    print('All tests passed!')