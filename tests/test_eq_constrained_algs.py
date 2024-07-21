# Test NewtonLagrange, and L2PenaltyEq

from all_problem_types import EqConstrained, eq_constrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal

def test_newton_lagrange(): 
    import numpy as np
    from modopt import NewtonLagrange

    prob = EqConstrained()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-12, 'feas_tol': 1e-12}
    optimizer = NewtonLagrange(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=11)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['c'], 0., decimal=11)
    assert_almost_equal(optimizer.results['pi'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 5
    assert optimizer.results['ngev'] == 3
    
    prob = eq_constrained_lite()
    prob.x0 = np.array([2., 2.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = NewtonLagrange(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=11)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['c'], 0., decimal=11)
    assert_almost_equal(optimizer.results['pi'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert_almost_equal(optimizer.results['feasibility'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 5
    assert optimizer.results['ngev'] == 3

def test_l2_penalty_eq():
    import numpy as np
    from modopt import L2PenaltyEq

    prob = EqConstrained()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-8, 'feas_tol': 1e-12, 'rho': 1000000.}
    optimizer = L2PenaltyEq(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=4)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=8)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 146
    assert optimizer.results['ngev'] == 64
    
    prob = eq_constrained_lite()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = L2PenaltyEq(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=4)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=8)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 146
    assert optimizer.results['ngev'] == 64

if __name__ == '__main__':
    test_newton_lagrange()
    test_l2_penalty_eq()
    print('All tests passed!')