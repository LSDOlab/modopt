# Test Jax interface

import pytest

@pytest.mark.interfaces
@pytest.mark.jax
def test_jax_problem():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import JaxProblem
    import jax
    import jax.numpy as jnp 
    jax.config.update("jax_enable_x64", True)

    # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

    jax_obj = lambda x: jnp.sum(x ** 4)
    jax_con = lambda x: jnp.array([x[0] + x[1], x[0] - x[1]])

    x0 = np.array([1., 2.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    x_sc = np.array([100., 0.2])
    o_sc = 3.
    c_sc = np.array([20., 5.])
    prob = JaxProblem(x0, name='quartic', jax_obj=jax_obj, jax_con=jax_con,
                      xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_sc, o_scaler=o_sc, c_scaler=c_sc)

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert_array_equal(prob.o_scaler, 3.)
    assert_array_equal(prob.x_scaler, [100., 0.2])
    assert_array_equal(prob.c_scaler, [20., 5.])
    assert_array_equal(prob.x0, np.array([100., 0.4]))
    assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 5.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SLSQP
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True})

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)

@pytest.mark.interfaces
@pytest.mark.jax
def test_problem():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import Problem
    import jax
    import jax.numpy as jnp 
    jax.config.update("jax_enable_x64", True)

    # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

    jax_obj = lambda x: jnp.sum(x ** 4)
    jax_con = lambda x: jnp.array([x[0] + x[1], x[0] - x[1]])

    _obj  = jax.jit(jax_obj)
    _grad = jax.jit(jax.grad(jax_obj))
    _con  = jax.jit(jax_con)
    _jac  = jax.jit(jax.jacfwd(jax_con))

    obj  = lambda x: np.float64(_obj(x))
    grad = lambda x: np.array(_grad(x))
    con  = lambda x: np.array(_con(x))
    jac  = lambda x: np.array(_jac(x))

    class Quartic(Problem):
        def initialize(self, ):
            self.problem_name = 'quartic'

        def setup(self):
            self.add_design_variables('x',
                                      shape=(2, ),
                                      lower=np.array([0., -np.inf]),
                                      upper=np.array([np.inf, np.inf]),
                                      vals=np.array([1., 2.]),
                                      scaler=np.array([100., 0.2]))

            self.add_objective('f', scaler=3.)

            self.add_constraints('c',
                                shape=(2, ),
                                lower=np.array([1., 1.]),
                                upper=np.array([1., np.inf]),
                                scaler=np.array([20., 5.]))

        def setup_derivatives(self):
            self.declare_objective_gradient(wrt='x', vals=None)
            jac_0 = jac(self.x0 / self.x_scaler) # Note that self.x0 is scaled
            self.declare_constraint_jacobian(of='c',
                                            wrt='x',
                                            vals=jac_0)

        def compute_objective(self, dvs, o):
            x = dvs['x']
            o['f'] = obj(x)

        def compute_objective_gradient(self, dvs, g):
            x = dvs['x']
            g['x'] = grad(x)

        def compute_constraints(self, dvs, c):
            x = dvs['x']
            c['c'] = con(x)

        def compute_constraint_jacobian(self, dvs, j):
            pass
            # x = dvs['x']
            # j['c', 'x'] = jac(x)

    prob = Quartic(jac_format='dense')

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert prob.options['jac_format'] == 'dense'
    assert_array_equal(prob.obj_scaler, {'f': 3.})
    assert_array_equal(prob.x_scaler, [100., 0.2])
    assert_array_equal(prob.c_scaler, [20., 5.])
    assert_array_equal(prob.x0, np.array([100., 0.4]))
    assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 5.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SQP, SLSQP, SNOPT, PySLSQP
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True})
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    # optimizer = SQP(prob, maxiter=20)
    snopt_options = {
        'Infinite bound': 1.0e20, 
        'Verify level': 3,
        'Verbose': True,
        }
    # optimizer = SNOPT(prob, solver_options=snopt_options)

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)
    # assert_almost_equal(optimizer.results['objective'], 3., decimal=11)

@pytest.mark.interfaces
@pytest.mark.jax
def test_problem_lite():
    import numpy as np
    from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
    from modopt import ProblemLite
    import jax
    import jax.numpy as jnp 
    jax.config.update("jax_enable_x64", True)

    # minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

    jax_obj = lambda x: jnp.sum(x ** 4)
    jax_con = lambda x: jnp.array([x[0] + x[1], x[0] - x[1]])

    _obj  = jax.jit(jax_obj)
    _grad = jax.jit(jax.grad(jax_obj))
    _con  = jax.jit(jax_con)
    _jac  = jax.jit(jax.jacfwd(jax_con))

    obj  = lambda x: np.float64(_obj(x))
    grad = lambda x: np.array(_grad(x))
    con  = lambda x: np.array(_con(x))
    jac  = lambda x: np.array(_jac(x))

    x0 = np.array([1., 2.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    x_sc = np.array([100., 0.2])
    o_sc = 3.
    c_sc = np.array([20., 5.])
    prob = ProblemLite(x0, name='quartic', obj=obj, grad=grad, con=con, jac=jac,
                       xl=xl, xu=xu, cl=cl, cu=cu, x_scaler=x_sc, o_scaler=o_sc, c_scaler=c_sc)

    assert prob.problem_name == 'quartic'
    assert prob.constrained == True
    assert_array_equal(prob.o_scaler, 3.)
    assert_array_equal(prob.x_scaler, [100., 0.2])
    assert_array_equal(prob.c_scaler, [20., 5.])
    assert_array_equal(prob.x0, np.array([100., 0.4]))
    assert_array_equal(prob.x_lower, np.array([0, -np.inf]))
    assert_array_equal(prob.x_upper, np.array([np.inf, np.inf]))
    assert_array_equal(prob.c_lower, [20., 5.])
    assert_array_equal(prob.c_upper, [20., np.inf])

    from modopt import SQP, SLSQP, SNOPT, PySLSQP
    optimizer = SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True})
    # optimizer = PySLSQP(prob, solver_options={'acc':1e-6, 'maxiter':20})
    # optimizer = SQP(prob, maxiter=20)
    snopt_options = {
        'Infinite bound': 1.0e20, 
        'Verify level': 3,
        'Verbose': True,
        }
    # optimizer = SNOPT(prob, solver_options=snopt_options)

    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)
    # assert_almost_equal(optimizer.results['objective'], 3., decimal=11)

if __name__ == '__main__':
    test_jax_problem()
    test_problem()
    test_problem_lite()
    print("All tests passed!")