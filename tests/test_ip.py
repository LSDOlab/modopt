# Test the InteriorPoint optimizer

from all_problem_types import Constrained, constrained_lite, Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

@pytest.mark.interior_point
def test_interior_point(): 
    import numpy as np
    from modopt import InteriorPoint

    prob = Constrained()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-8, 'feas_tol': 1e-8}
    optimizer = InteriorPoint(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0) # Note: this adds 3 to obj/con_evals and 1 to grad/jac_evals
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=9)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=8)
    assert_array_almost_equal(optimizer.results['c'], [0., 1.,  0.], decimal=8)
    assert_array_almost_equal(optimizer.results['pi'], [1.33333333, 0., 1.33333333], decimal=8)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=9)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=9)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 10
    assert optimizer.results['ngev'] == 10
    
    prob = constrained_lite()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = InteriorPoint(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0) # Note: this adds 3 to obj/con_evals and 1 to grad/jac_evals
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=9)
    assert_almost_equal(optimizer.results['objective'], 1., decimal=8)
    assert_array_almost_equal(optimizer.results['c'], [0., 1.,  0.], decimal=8)
    assert_array_almost_equal(optimizer.results['pi'], [1.33333333, 0., 1.33333333], decimal=8)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=9)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=9)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 10
    assert optimizer.results['ngev'] == 10

    prob = Unconstrained()
    prob.x0 = np.array([1., 1.])

    optimizer = InteriorPoint(prob, **solver_options) 
    optimizer.check_first_derivatives(prob.x0) # Note: this adds 3 to obj_evals and 1 to grad_evals
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=12)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=12)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=12)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 2
    assert optimizer.results['ngev'] == 2

    prob = unconstrained_lite()
    prob.x0 = np.array([1., 1.])

    optimizer = InteriorPoint(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0) # Note: this adds 3 to obj_evals and 1 to grad_evals
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['success']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=12)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=12)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=12)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 2
    assert optimizer.results['ngev'] == 2

if __name__ == '__main__':
    test_interior_point()
    print('All tests passed!')