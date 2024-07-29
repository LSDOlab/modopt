# Test casadi interface - same example as csdl

import pytest
import modopt as mo
import numpy as np
import casadi as ca
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal

# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

@pytest.mark.interfaces
@pytest.mark.casadi
def test_casadi_problem():
    # METHOD 1: Use CasADi expressions directly in mo.CasadiProblem.
    #           ModOpt will auto-generate the gradient, Jacobian, and objective Hessian.
    #           ModOpt will also auto-generate the Lagrangian, its gradient, and Hessian.
    #           No need to manually generate functions or their derivatives and then wrap them.

    obj = lambda x: ca.sum1(x**4)
    con = lambda x: ca.vertcat(x[0] + x[1], x[0] - x[1])

    prob = mo.CasadiProblem(x0=np.array([1., 2.]), ca_obj=obj, ca_con=con, 
                            xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
                            cl=np.array([1., 1.]), cu=np.array([1., np.inf]), name='quartic_casadi',
                            x_scaler=np.array([100., 0.2]), c_scaler=np.array([20., 5.]), o_scaler=3.)
    
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)


@pytest.mark.interfaces
@pytest.mark.casadi
def test_problem_lite_casadi():
    # METHOD 2: Create CasADi functions and derivatives, 
    #           and wrap them manually before passing to Problem/ProblemLite.

    # Create scalar/matrix symbols
    x = ca.MX.sym('x', 2)

    # Compose into expressions to evaluate objective
    obj_expr = ca.sum1(x**4)

    # Sensitivity of expression -> new expression
    grad_expr = ca.gradient(obj_expr,x)

    # Create a Function to evaluate obj and grad expressions
    _obj  = ca.Function('o',[x],[obj_expr])
    _grad = ca.Function('g',[x],[grad_expr])

    # Create an expression for the constraints and their jacobian
    con_expr = ca.vertcat(x[0] + x[1], x[0] - x[1])
    jac_expr = ca.jacobian(con_expr, x)

    # Create a Function to evaluate con and jac expressions
    _con = ca.Function('c', [x], [con_expr])
    _jac = ca.Function('j', [x], [jac_expr])

    # Wrap the functions for modopt
    obj  = lambda x: np.float64(_obj(x))
    grad = lambda x: np.array(_grad(x)).flatten()
    con  = lambda x: np.array(_con(x)).flatten()
    jac  = lambda x: np.array(_jac(x))

   # Create a ProblemLite object using the wrapped functions
    prob = mo.ProblemLite(x0=np.array([1., 2.]), obj=obj, grad=grad, con=con, jac=jac,
                          xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
                          cl=np.array([1., 1.]), cu=np.array([1., np.inf]),
                          name = 'quartic_casadi',
                          x_scaler=np.array([100., 0.2]), c_scaler=np.array([20., 5.]), o_scaler=3.)
    
    optimizer = mo.SLSQP(prob, solver_options={'ftol': 1e-6, 'maxiter': 20, 'disp': True})
    optimizer.check_first_derivatives(prob.x0)
    optimizer.solve()
    optimizer.print_results()

    assert optimizer.results['success'] == True
    assert optimizer.results['message'] == 'Optimization terminated successfully'
    assert_array_almost_equal(optimizer.results['x'], [100., 0.], decimal=11)
    assert_almost_equal(optimizer.results['fun'], 3., decimal=11)

if __name__ == '__main__':
    test_casadi_problem()
    test_problem_lite_casadi()
    print("All tests passed!")