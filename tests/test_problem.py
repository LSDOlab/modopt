# Test the ProblemLite class
from modopt import Problem
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest
from all_problem_types import Scaling

def test_callback_list():
    x0 = np.array([1., 1.])
    prob = Scaling()

    assert prob.nx == 2
    assert prob.nc == 2
    assert set(prob.declared_variables) == {'dv', 'obj', 'con', 'grad', 'jac'}
    assert set(prob.user_defined_callbacks) == {'obj', 'con', 'grad', 'jac'}

def test_errors():
    class TestProblem0(Problem):
        def initialize(self):
            self.problem_name = 'test0'
        def setup(self):
            pass
        def setup_derivatives(self):
            pass

    with pytest.raises(Exception) as excinfo:
        prob = TestProblem0()
    print(str(excinfo.value))
    assert excinfo.type is Exception
    assert str(excinfo.value) == 'No design variables are declared.'

    class TestProblem1(Problem):
        def initialize(self):
            self.problem_name = 'test1'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]))
        def setup_derivatives(self):
            pass

    with pytest.raises(Exception) as excinfo:
        prob = TestProblem1()
    assert excinfo.type is Exception
    assert str(excinfo.value) == 'No objective or constraints are declared.'

    class TestProblem2(Problem):
        def initialize(self):
            self.problem_name = 'test2'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]))
            self.add_objective('f', scaler=2.)
        def setup_derivatives(self):
            pass
        
    with pytest.raises(Exception) as excinfo:
        prob = TestProblem2()
    assert excinfo.type is Exception
    assert str(excinfo.value) == 'Objective is declared but compute_objective() method is not implemented.'

    class TestProblem3(Problem):
        def initialize(self):
            self.problem_name = 'test3'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]))
            self.add_constraints('c', shape=(2,), lower=1.)
        def setup_derivatives(self):
            pass

    with pytest.raises(Exception) as excinfo:
        prob = TestProblem3()
    assert excinfo.type is Exception
    assert str(excinfo.value) == 'Constraints are declared but compute_constraints() method is not implemented.'

    class TestProblem4(Problem):
        def initialize(self):
            self.problem_name = 'test4'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]))
            self.add_objective('f', scaler=2.)
        def setup_derivatives(self):
            self.declare_objective_gradient(wrt='x')
        def compute_objective(self, dvs, obj):
            obj['f'] = np.sum(dvs['x']**2)

    with pytest.raises(Exception) as excinfo:
        prob = TestProblem4()
    assert excinfo.type is Exception
    assert str(excinfo.value) == "Objective gradient is declared but compute_objective_gradient() method is not implemented."\
                                "If declared derivatives are constant, define an empty compute_objective_gradient() with 'pass'."\
                                "If declared derivatives are not available, define a compute_objective_gradient() method"\
                                "that calls self.use_finite_differencing('objective_gradient', step=1.e-6)."\
                                "If using a gradient-free optimizer, do not declare objective gradient."
    
    class TestProblem5(Problem):
        def initialize(self):
            self.problem_name = 'test5'
        
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]))
            self.add_constraints('c', shape=(2,), lower=1.)

        def setup_derivatives(self):
            self.declare_constraint_jacobian(of='c', wrt='x')
        
        def compute_constraints(self, dvs, con):
            con['c'] = dvs['x']**2 - 1.
        
    with pytest.raises(Exception) as excinfo:
        prob = TestProblem5()
    assert excinfo.type is Exception
    assert str(excinfo.value) == "Constraint Jacobian is declared but compute_constraint_jacobian() method is not implemented."\
                                "If declared derivatives are constant, define an empty compute_constraint_jacobian() with 'pass'."\
                                "If declared derivatives are not available, define a compute_constraint_jacobian() method"\
                                "that calls self.use_finite_differencing('constraint_jacobian', step=1.e-6)."\
                                "If using a gradient-free optimizer, do not declare constraint Jacobian."

# TODO: Add more tests for the following methods
# def test_declare_errors():
#     pass

# def test_lagrangian_computations():
#     pass

# def test_finite_differencing():
#     pass

# def test_vector_products():
#     pass

# def test_unavailable_vector_products():
#     pass

def test_all_dummy_methods():
    # A gradient-free unconstrained problem
    class TestProblem1(Problem):
        def initialize(self):
            self.problem_name = 'test1'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]), lower=-1e99, upper=1e99)
            self.add_objective('f', scaler=2.)
        def setup_derivatives(self):
            pass
        def compute_objective(self, dvs, obj):
            obj['f'] = np.sum(dvs['x']**2)

    prob = TestProblem1()

    COMPUTE_NOT_IMPLEMENTED_ERROR = r".*\(\) method is not implemented by the user in the derived class TestProblem1."

    # compute_objective() is always implemented,
    # as feasibility problems will have a dummy objective function,
    # and a dummy gradient function if the problem is gradient-based.

    dummy_funcs_2 = [
        # prob.compute_objective,  # compute_objective() is implemented
        prob.compute_objective_gradient,
        prob.compute_objective_hessian,

        prob.compute_constraints,
        prob.compute_constraint_jacobian,
    ]

    dummy_funcs_3 = [
        prob.compute_objective_hvp,

        prob.compute_lagrangian,
        prob.compute_lagrangian_gradient,
        prob.compute_lagrangian_hessian,

        prob.compute_constraint_jvp,
        prob.compute_constraint_vjp
    ]
    
    # test dummy methods with 2 arguments
    for func in dummy_funcs_2:
        with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
            func(1, 2)

    # test dummy methods with 3 arguments
    for func in dummy_funcs_3:
        with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
            func(1, 2, 3)
    
    # test dummy methods with 4 arguments
    COMPUTE_NOT_IMPLEMENTED_ERROR = r"compute_lagrangian_hvp\(\) method is not implemented by the user in the derived class TestProblem1."
    with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
        prob.compute_lagrangian_hvp(1, 2, 3, 4)

    # A gradient-free feasibility problem
    class TestProblem2(Problem):
        def initialize(self):
            self.problem_name = 'test2'
        def setup(self):
            self.add_design_variables('x', shape=(2,), vals=np.array([1., 2.]), lower=-1e99, upper=1e99)
            self.add_constraints('c', shape=(2,), lower=1.)
        def setup_derivatives(self):
            pass
        def compute_constraints(self, dvs, con):
            con['c'] = dvs['x']**2 - 1.
    
    prob = TestProblem2()

    COMPUTE_NOT_IMPLEMENTED_ERROR = r".*\(\) method is not implemented by the user in the derived class TestProblem2."

    dummy_funcs_2 = [
        # prob.compute_objective,  # dummy compute_objective() is implemented
        prob.compute_objective_gradient,
        prob.compute_objective_hessian,

        # prob.compute_constraints, # compute_constraints() is implemented
        prob.compute_constraint_jacobian,
    ]

    dummy_funcs_3 = [
        prob.compute_objective_hvp,

        prob.compute_lagrangian,
        prob.compute_lagrangian_gradient,
        prob.compute_lagrangian_hessian,

        prob.compute_constraint_jvp,
        prob.compute_constraint_vjp
    ]

    assert prob.compute_objective(1, 2) is None
    # test dummy methods with 2 arguments
    for func in dummy_funcs_2:
        with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
            func(1, 2)

    # test dummy methods with 3 arguments
    for func in dummy_funcs_3:
        with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
            func(1, 2, 3)
    
    # test dummy methods with 4 arguments
    COMPUTE_NOT_IMPLEMENTED_ERROR = r"compute_lagrangian_hvp\(\) method is not implemented by the user in the derived class TestProblem2."
    with pytest.raises(NotImplementedError, match=COMPUTE_NOT_IMPLEMENTED_ERROR):
        prob.compute_lagrangian_hvp(1, 2, 3, 4)

def test_str():
    prob = Scaling()

    assert str(prob) == \
    "\n"\
    "\tProblem Overview:\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\tProblem name             : scaling\n"\
    "\tObjectives               : f\n"\
    "\tDesign variables         : x (2,)\n"\
    "\tConstraints              : c (2,)\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\n"\
    "\tProblem Data (UNSCALED):\n"\
    "\t----------------------------------------------------------------------------------------------------\n"\
    "\tObjectives:\n"\
    "\tIndex | Name       | Scaler        | Value         \n"\
    "\t    0 | f          | +2.000000e+01 | +1.000000e+00\n"\
    "\n"\
    "\tDesign Variables:\n"\
    "\tIndex | Name       | Scaler        | Lower Limit   | Value         | Upper Limit   \n"\
    "\t    0 | x[0]       | +2.000000e+00 | +0.000000e+00 | +5.000000e+01 | +1.000000e+99\n"\
    "\t    1 | x[1]       | +2.000000e-01 | -1.000000e+99 | +5.000000e+00 | +1.000000e+99\n"\
    "\n"\
    "\tConstraints:\n"\
    "\tIndex | Name       | Scaler        | Lower Limit   | Value         | Upper Limit   | Lag. mult.    \n"\
    "\t    0 | c[0]       | +5.000000e+00 | +1.000000e+00 | +0.000000e+00 | +1.000000e+00 | \n"\
    "\t    1 | c[1]       | +5.000000e-01 | +1.000000e+00 | +0.000000e+00 | +1.000000e+99 | \n"\
    "\t----------------------------------------------------------------------------------------------------\n"

if __name__ == '__main__':
    test_errors()
    # test_declare_errors()
    # test_finite_differencing()
    # test_lagrangian_computations()
    # test_vector_products()
    # test_unavailable_vector_products()
    test_all_dummy_methods()
    test_str()

    print('All tests passed!')