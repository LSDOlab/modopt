# Test SteepestDescent, Newton, and QuasiNewton

from all_problem_types import Unconstrained, unconstrained_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

def test_steeepest_descent():
    import numpy as np
    from modopt import SteepestDescent

    prob = Unconstrained()
    prob.x0 = np.array([1., 1.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-4}
    optimizer = SteepestDescent(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=6)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=4)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 98
    assert optimizer.results['ngev'] == 96
    
    prob = unconstrained_lite()
    prob.x0 = np.array([1., 1.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = SteepestDescent(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=6)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=4)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 98
    assert optimizer.results['ngev'] == 96

def test_quasi_newton():
    import numpy as np
    from modopt import QuasiNewton

    prob = Unconstrained()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-11}
    optimizer = QuasiNewton(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=4)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 76
    assert optimizer.results['ngev'] == 76
    
    prob = unconstrained_lite()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = QuasiNewton(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=4)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 76
    assert optimizer.results['ngev'] == 76

from modopt import Problem, ProblemLite
import numpy as np

class Ord2Unconstrained(Problem):
    def initialize(self):
        self.problem_name = 'unconstrained2'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([500., 5.]))
        self.add_objective('f')

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_objective_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_objective_hessian(self, dvs, hess):
        hess['x','x'] = np.diag(12 * dvs['x'] ** 2)

def ord2_unconstrained_lite():
    x0 = np.array([500., 5.])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def obj_hess(x):
        return np.diag(12 * x ** 2)
    
    return ProblemLite(x0, obj=obj, grad=grad, obj_hess=obj_hess, name='unconstrained2_lite')

def test_newton():
    import numpy as np
    from modopt import Newton

    prob = Ord2Unconstrained()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-15}
    optimizer = Newton(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=5)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 42
    assert optimizer.results['ngev'] == 83
    

    prob = ord2_unconstrained_lite()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    optimizer = Newton(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)
    
    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=5)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 42
    assert optimizer.results['ngev'] == 83

def test_no_ls_algs():
    from modopt.core.optimization_algorithms.steepest_descent_no_ls import SteepestDescentNoLS
    from modopt.core.optimization_algorithms.quasi_newton_no_ls import QuasiNewtonNoLS
    from modopt.core.optimization_algorithms.newton_no_ls import NewtonNoLS

    prob = ord2_unconstrained_lite()
    prob.x0 = np.array([0.7, 0.7]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-3}
    optimizer = SteepestDescentNoLS(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=1)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=4)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=3)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 29
    assert optimizer.results['ngev'] == 29

    prob = ord2_unconstrained_lite()
    prob.x0 = np.array([10., 10.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-11}
    optimizer = QuasiNewtonNoLS(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=4)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 45
    assert optimizer.results['ngev'] == 45

    prob = ord2_unconstrained_lite()
    prob.x0 = np.array([100., 10.]) # set initial guess to something closer to the minimum [0, 0]

    solver_options = {'maxiter': 100, 'opt_tol': 1e-15}
    optimizer = NewtonNoLS(prob, **solver_options)
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results(summary_table=True)

    assert optimizer.results['converged']
    assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=5)
    assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
    assert_almost_equal(optimizer.results['optimality'], 0., decimal=11)
    assert optimizer.results['niter'] < solver_options['maxiter']
    assert optimizer.results['nfev'] == 42
    assert optimizer.results['ngev'] == 42

if __name__ == '__main__':
    test_steeepest_descent()
    test_quasi_newton()
    test_newton()
    test_no_ls_algs()
    print('All tests passed!')