import numpy as np
from modopt import Problem, ProblemLite

class BeanFunction(Problem):
    # Global minimum: f(x) = 0.09194 at x = [1.21314, 0.82414]
    def initialize(self, ):
        self.problem_name = 'bean_function'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                #   vals=np.array([-1,1.5]),
                                  vals=np.zeros(2,),
                                  lower=-10*np.ones(2,),
                                  upper=10*np.ones(2,))
        self.add_objective('f')

    def compute_objective(self, dvs, obj):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        obj['f'] = (1-x1)**2 + (1-x2)**2 + 0.5*(2*x2 - x1**2)**2

def bean_function_lite():
    x0 = np.array([0., 0.])
    xl = -np.array([10, 10])
    xu =  np.array([10, 10])
    obj = lambda x: (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2
    return ProblemLite(x0, xl=xl, xu=xu, obj=obj, name='bean_function_lite', grad_free=True)

def test_pso():
    '''
    Test the PSO algorithm for unconstrained problems.
    '''
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import PSO

    prob = BeanFunction()
    # Empirically arrived best hyperparameter values for BeanFunction for tol=1e-4:
    # population=20, w = 0.5, c_g = 0.3, c_p = 0.4
    solver_options = {
        'tol': 1e-4,
        'maxiter': 500,
        'population': 20,
        'inertia_weight': 0.5,      # w
        'cognitive_coeff': 0.4,     # c_g
        'social_coeff': 0.3,        # c_p
        }

    optimizer = PSO(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_almost_equal(optimizer.results['x'], [1.21314, 0.82414], decimal=2)
    assert_almost_equal(optimizer.results['objective'], 0.09194, decimal=5)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']

    prob = bean_function_lite()
    optimizer = PSO(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_almost_equal(optimizer.results['x'], [1.21314, 0.82414], decimal=2)
    assert_almost_equal(optimizer.results['objective'], 0.09194, decimal=5)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']

def test_nelder_mead():
    '''
    Test the Nelder-Mead algorithm for unconstrained problems.
    '''
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import NelderMeadSimplex

    prob = BeanFunction()
    solver_options = {
        'maxiter': 200,
        'tol': 1e-6,
        'initial_length': 1.0, 
        }
    optimizer = NelderMeadSimplex(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_almost_equal(optimizer.results['x'], [1.21314, 0.82414], decimal=3)
    assert_almost_equal(optimizer.results['objective'], 0.09194, decimal=5)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']

    prob = bean_function_lite()
    optimizer = NelderMeadSimplex(prob, **solver_options)
    optimizer.solve()
    print(optimizer.results)
    optimizer.print_results()

    assert_array_almost_equal(optimizer.results['x'], [1.21314, 0.82414], decimal=3)
    assert_almost_equal(optimizer.results['objective'], 0.09194, decimal=5)
    assert optimizer.results['converged']
    assert optimizer.results['niter'] <= solver_options['maxiter']

if __name__ == '__main__':
    test_pso()
    test_nelder_mead()
    print("All tests passed!")