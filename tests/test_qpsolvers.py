# Test the qpsolvers interface

from all_problem_types import ConvexQP, convex_qp_lite
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

@pytest.mark.interfaces
@pytest.mark.qpsolvers
def test_qpsolvers(): 
    from modopt import ConvexQPSolvers

    probs = [ConvexQP(), convex_qp_lite()]
    solver_options = {'solver':'quadprog', 'verbose':True}

    for prob in probs:
        optimizer = ConvexQPSolvers(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_dual_variables = True, 
                                optimal_constraints = True,  
                                optimal_variables = True,  
                                extras = True)

        assert optimizer.results['found']
        assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)
        assert_array_almost_equal(optimizer.results['z_box'], [0., 0.], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 1., decimal=11)
        assert_almost_equal(optimizer.results['primal_residual'], 4.44e-16, decimal=11)
        assert_almost_equal(optimizer.results['dual_residual'], 0., decimal=11)
        assert_almost_equal(optimizer.results['duality_gap'], 4.44e-16, decimal=11)
        assert_array_almost_equal(optimizer.results['constraints'], [1., 1.], decimal=11)
        assert_array_almost_equal(optimizer.results['y'], [-1.], decimal=11)    # dual variables for the equality constraints
        assert_array_almost_equal(optimizer.results['z'], [1.], decimal=11)     # dual variables for the inequality constraints
        assert_array_almost_equal(optimizer.results['extras']['iact'], [1, 2])
        assert_array_almost_equal(optimizer.results['extras']['iterations'], [3, 0])

@pytest.mark.interfaces
@pytest.mark.qpsolvers
def test_errors():
    import numpy as np
    from modopt import Problem, ProblemLite, ConvexQPSolvers

    # 1. raise error when no solver is specified
    probs = [ConvexQP(), convex_qp_lite()]
    for prob in probs:
        with pytest.raises(ValueError) as excinfo:
            optimizer = ConvexQPSolvers(prob, solver_options={ 'verbose':False})
        assert "Please specify a 'solver' in the 'solver_options' dictionary. Available solvers are: " in str(excinfo.value)

    class QPNoHess(Problem):
        def setup(self):
            self.add_design_variables('x', shape=(2,))
            self.add_constraints('c', shape=(1,))
            self.add_objective('f')
        def setup_derivatives(self):
            self.declare_objective_gradient(wrt='x', vals=None)
            self.declare_constraint_jacobian(of='c', wrt='x', vals=np.array([[1.,1.]]))

        def compute_objective(self, dvs, obj):
            obj['f'] = np.sum(dvs['x']**2)
        def compute_objective_gradient(self, dvs, grad):
            grad['x'] = 2 * dvs['x']
        def compute_constraints(self, dvs, cons):
            cons['c'][0] = dvs['x'][0] + dvs['x'][1]
        def compute_constraint_jacobian(self, dvs, jac):
            pass

       
    solver_options = {'solver':'quadprog', 'verbose':True}

    # 2. Raise error when objective Hessian is not declared for Problem() class
    with pytest.raises(ValueError) as excinfo:
        optimizer = ConvexQPSolvers(QPNoHess(), solver_options=solver_options)
    assert str(excinfo.value) == "Objective Hessian function is not declared in the Problem() subclass but is needed for ConvexQPSolvers."
    
    def qp_no_hess_lite():
        x0 = np.array([500., 5.])
        cl = np.array([1., 1.])
        cu = np.array([1., np.inf])
        xl = np.array([0., -np.inf])
        xu = np.array([np.inf, np.inf])
        def obj(x):
            return np.sum(x**2)
        def grad(x):    
            return 2 * x
        def con(x):
            return np.array([x[0] + x[1], x[0] - x[1]])
        def jac(x):
            return np.array([[1., 1.], [1., -1]])
        
        return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac,
                           xl=xl, xu=xu, cl=cl, cu=cu, name='qp_no_hess_lite')
    
    # 3. Raise warning when objective Hessian is not declared for ProblemLite() class and use finite-differences
    optimizer = ConvexQPSolvers(qp_no_hess_lite(), solver_options=solver_options)
    optimizer.solve()
    optimizer.print_results()
    assert optimizer.results['found']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=11)

    class QPNoJac(Problem):
        def setup(self):
            self.add_design_variables('x', shape=(2,))
            self.add_constraints('c', shape=(1,))
            self.add_objective('f')
        def setup_derivatives(self):
            self.declare_objective_gradient(wrt='x', vals=None)
            self.declare_objective_hessian(of='x', wrt='x', vals=2*np.eye(2))

        def compute_objective(self, dvs, obj):
            obj['f'] = np.sum(dvs['x']**2)
        def compute_objective_gradient(self, dvs, grad):
            grad['x'] = 2 * dvs['x']
        def compute_constraints(self, dvs, cons):
            cons['c'][0] = dvs['x'][0] + dvs['x'][1]
        def compute_objective_hessian(self, dvs, hess):
            pass
            
    # 4. Raise error when constraint Jacobian is not declared for Problem() class
    with pytest.raises(ValueError) as excinfo:
        optimizer = ConvexQPSolvers(QPNoJac(), solver_options=solver_options)
    assert str(excinfo.value) == "Constraint Jacobian function is not declared in the Problem() subclass but is needed for ConvexQPSolvers."

    def qp_no_jac_lite():
        x0 = np.array([500., 5.])
        cl = np.array([1., 1.])
        cu = np.array([1., np.inf])
        xl = np.array([0., -np.inf])
        xu = np.array([np.inf, np.inf])
        def obj(x):
            return np.sum(x**2)
        def grad(x):    
            return 2 * x
        def obj_hess(x):
            return 2 * np.eye(2)
        def con(x):
            return np.array([x[0] + x[1], x[0] - x[1]])
        
        return ProblemLite(x0, obj=obj, grad=grad, con=con, obj_hess=obj_hess,
                           xl=xl, xu=xu, cl=cl, cu=cu, name='qp_no_jac_lite')

    # 5. Raise warning when constraint Jacobian is not declared for ProblemLite() class and use finite-differences
    optimizer = ConvexQPSolvers(qp_no_jac_lite(), solver_options=solver_options)
    optimizer.solve()
    optimizer.print_results()
    assert optimizer.results['found']
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=6)

if __name__ == '__main__':
    test_qpsolvers()
    test_errors()
    print('All tests passed!')