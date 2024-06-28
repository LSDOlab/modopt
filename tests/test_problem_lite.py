# Test the ProblemLite class

from modopt import ProblemLite
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

def test_type_errors():
    x0 = [1., 1.]
    obj = lambda x: np.array([np.sum(x**2)])
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj)
    assert excinfo.type is TypeError
    assert str(excinfo.value) == 'Initial guess x0 must be a numpy array.'

    x0 = np.array([1., 1.])
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, name=12, obj=obj)
    assert excinfo.type is TypeError
    assert str(excinfo.value) == 'Problem "name" must be a string.'

    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, name="test_prob", obj=obj, grad_free=12)
    assert excinfo.type is TypeError
    assert str(excinfo.value) == '"grad_free" argument must be a boolean.'

    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj="test")
    assert excinfo.type is TypeError
    assert str(excinfo.value) == 'Objective function "obj" must be a callable function.'

    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj, vp_fd_step="test")
    assert excinfo.type is TypeError
    assert str(excinfo.value) == 'Vector product finite difference step "vp_fd_step" must be a real-valued scalar.'

    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj, fd_step="test")
    assert excinfo.type is TypeError
    assert str(excinfo.value) == 'Finite difference step "fd_step" must be a real-valued scalar or a numpy array.'

def test_callback_list():
    x0 = np.array([1., 1.])
    obj = lambda x: np.sum(x**2)
    con = lambda x: x**2 - 1.
    prob = ProblemLite(x0, obj=obj, con=con)

    assert prob.nx == 2
    assert prob.nc == 2
    assert set(prob.user_defined_callbacks) == {'obj', 'con'}

def test_callback_errors():
    x0 = np.array([1., 1.])
    obj = lambda x: x**2
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'Objective function "obj" must return a scalar or a 1D array with shape (1,).'

    obj = lambda x: "test"
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'Objective function "obj" must return a real-valued scalar.'

    obj = lambda x: np.sum(x**2)
    grad = lambda x: 1.
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj, grad=grad)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'Gradient function "grad" must return a 1D array with shape (nx,), ' \
                                 'where nx=2 is the number of design variables.'   

    con = lambda x: (x**2 - 1).reshape(2, 1)
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, con=con)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'Constraint function "con" must return a 1D array with shape (nc,) ' \
                                 'where nc=2 is the number of constraints.'

    con = lambda x: x**2 - 1
    jac = lambda x: 1.
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, con=con, jac=jac)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'Jacobian function "jac" must return a 2D array with shape (nc, nx), ' \
                                 'where nc=2 is the number of constraints and nx=2 is the number of design variables.'

def test_check_shapes():
    x0 = np.array([])
    obj = lambda x: np.sum(x**2)
    with pytest.raises(Exception) as excinfo:
        prob = ProblemLite(x0, obj=obj)
    assert excinfo.type is ValueError
    assert str(excinfo.value) == 'No design variables declared. "x0" has size 0. Please provide a non-empty initial guess.'

    # TODO: Add more tests for checking shapes of design variables, objectives, constraints, scalars, bounds, etc.

def test_lagrangian_computations():
    x0 = np.array([1., 1.])
    obj  = lambda x: np.sum(x**2)
    grad = lambda x: 2*x
    con = lambda x: x**2 + 1.
    jac = lambda x: np.diag(2*x)
    prob = ProblemLite(x0, obj=obj, con=con)

    mu = np.array([1., 2.])

    lag = prob._compute_lagrangian(prob.x0, mu)
    assert_almost_equal(lag, 8.)

    lag_grad = prob._compute_lagrangian_gradient(prob.x0, mu)
    assert_array_almost_equal(lag_grad, np.array([4., 6.]), decimal=5)

    lag_hess = prob._compute_lagrangian_hessian(prob.x0, mu)
    assert_array_almost_equal(lag_hess, np.array([[4., 0.], [0., 6.]]), decimal=3)

def test_vector_products():
    x0 = np.array([1., 1.])
    obj  = lambda x: np.sum(x**2)
    obj_hvp = lambda x, v: 2 * v
    lag_hvp = lambda x, mu, v: 2 * np.array([1 + mu[0], 1 + mu[1]]) * v
    con = lambda x: x**2 + 1.
    jvp = lambda x, v: 2 * x * v
    vjp = lambda x, v: 2 * x * v
    prob = ProblemLite(x0, obj=obj, obj_hvp=obj_hvp, lag_hvp=lag_hvp, con=con, jvp=jvp, vjp=vjp)

    mu = np.array([1., 2.])
    v  = np.array([2., 1.])

    obj_hvp = prob._compute_objective_hvp(prob.x0, v)
    assert_array_almost_equal(obj_hvp, np.array([4., 2.]), decimal=11)

    lag_hvp = prob._compute_lagrangian_hvp(prob.x0, mu, v)
    assert_array_almost_equal(lag_hvp, np.array([8., 6.]), decimal=11)

    jvp = prob._compute_constraint_jvp(prob.x0, v)
    assert_array_almost_equal(jvp, np.array([4., 2.]), decimal=11)

    vjp = prob._compute_constraint_vjp(prob.x0, v)
    assert_array_almost_equal(vjp, np.array([4., 2.]), decimal=11)

def test_unavailable_vector_products():
    x0 = np.array([1., 1.])
    obj  = lambda x: np.sum(x**2)
    con = lambda x: x**2 + 1.
    prob = ProblemLite(x0, obj=obj, con=con)

    mu = np.array([1., 2.])
    v  = np.array([2., 1.])

    obj_hvp = prob._compute_objective_hvp(prob.x0, v)
    assert_array_almost_equal(obj_hvp, np.array([4., 2.]), decimal=3)

    lag_hvp = prob._compute_lagrangian_hvp(prob.x0, mu, v)
    assert_array_almost_equal(lag_hvp, np.array([8., 6.]), decimal=3)

    jvp = prob._compute_constraint_jvp(prob.x0, v)
    assert_array_almost_equal(jvp, np.array([4., 2.]), decimal=5)

    vjp = prob._compute_constraint_vjp(prob.x0, v)
    assert_array_almost_equal(vjp, np.array([4., 2.]), decimal=5)

def test_all_dummy_methods():
    x0 = np.array([1., 1.])
    obj  = lambda x: np.sum(x**2)
    prob = ProblemLite(x0, obj=obj)

    assert prob.compute_objective(1, 2) is None
    assert prob.compute_objective_gradient(1, 2) is None
    assert prob.compute_objective_hessian(1, 2) is None
    assert prob.compute_objective_hvp(1, 2, 3) is None

    assert prob.compute_lagrangian(1, 2, 3) is None
    assert prob.compute_lagrangian_gradient(1, 2, 3) is None
    assert prob.compute_lagrangian_hessian(1, 2, 3) is None
    assert prob.compute_lagrangian_hvp(1, 2, 3, 4) is None

    assert prob.compute_constraints(1, 2) is None
    assert prob.compute_constraint_jacobian(1, 2) is None
    assert prob.compute_constraint_jvp(1, 2, 3) is None
    assert prob.compute_constraint_vjp(1, 2, 3) is None

def test_str():
    x0 = np.array([1., 1.])
    obj = lambda x: np.sum(x**2)
    con = lambda x: x**2 + 1.
    prob = ProblemLite(x0, obj=obj, con=con)

    mu = np.array([1., 2.])
    v  = np.array([2., 1.])
    lag_hvp = prob._compute_lagrangian_hvp(prob.x0, mu, v)
    assert_array_almost_equal(lag_hvp, np.array([8., 6.]), decimal=3)

    assert str(prob) == \
    "\n"\
    "\tProblem Overview:\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\tProblem name             : unnamed_problem\n"\
    "\tObjectives               : obj\n"\
    "\tDesign variables         : x   (shape: (2,))\n"\
    "\tConstraints              : con (shape: (2,))\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\n"\
    "\tProblem Data (UNSCALED):\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\tObjectives:\n"\
    "\tIndex | Name       | Scaler        | Value         \n"\
    "\t    0 | obj        | +1.000000e+00 | +2.000000e+00\n"\
    "\n"\
    "\tDesign Variables:\n"\
    "\tIndex | Name       | Scaler        | Lower Limit   | Value         | Upper Limit   \n"\
    "\t    0 | x[0]       | +1.000000e+00 | -1.000000e+99 | +1.000000e+00 | +1.000000e+99\n"\
    "\t    1 | x[1]       | +1.000000e+00 | -1.000000e+99 | +1.000000e+00 | +1.000000e+99\n"\
    "\n"\
    "\tConstraints:\n"\
    "\tIndex | Name       | Scaler        | Lower Limit   | Value         | Upper Limit   | Lag. mult.    \n"\
    "\t    0 | con[0]     | +1.000000e+00 | -1.000000e+99 | +2.000000e+00 | +1.000000e+99 | +1.000000e+00 \n"\
    "\t    1 | con[1]     | +1.000000e+00 | -1.000000e+99 | +2.000000e+00 | +1.000000e+99 | +2.000000e+00 \n"\
    "\t----------------------------------------------------------------------------------------------------\n"


if __name__ == '__main__':
    test_type_errors()
    test_callback_errors()
    test_check_shapes()
    test_lagrangian_computations()
    test_vector_products()
    test_unavailable_vector_products()
    test_all_dummy_methods()
    test_str()

    print('All tests passed!')