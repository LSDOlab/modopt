# Test the cvxopt interface
from modopt import Problem, ProblemLite
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_almost_equal
import pytest

class UnconstrainedConvex(Problem):
    def initialize(self):
        self.problem_name = 'unconstrained_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]))
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_objective_hessian(of='x', wrt='x', vals=2*np.eye(2))

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_objective_hessian(self, dvs, hess):
        pass
    
def unconstrained_convex_lite():
    x0 = np.array([500., 5.])
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def obj_hess(x):
        return 2 * np.eye(2)
    return ProblemLite(x0, obj=obj, grad=grad, obj_hess=obj_hess, 
                       name='unconstrained_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_unconstrained():
    from modopt import CVXOPT

    probs = [UnconstrainedConvex(), unconstrained_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
        assert optimizer.results['constraints'] == []
        assert_almost_equal(optimizer.results['primal objective'], 0., decimal=6)
        assert_almost_equal(optimizer.results['dual objective'], 0., decimal=11)
        assert_almost_equal(optimizer.results['gap'], 0., decimal=11)
        assert_almost_equal(optimizer.results['relative gap'], 0., decimal=6)
        assert_almost_equal(optimizer.results['primal infeasibility'], 0., decimal=11)
        assert_almost_equal(optimizer.results['dual infeasibility'], 0., decimal=11)
        assert_almost_equal(optimizer.results['primal slack'], 0., decimal=11)
        assert_almost_equal(optimizer.results['dual slack'], 1., decimal=11)

class BoundedConvex(Problem):
    def initialize(self):
        self.problem_name = 'bounded_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]),
                                  lower=np.array([0., -np.inf]), upper=np.array([np.inf, np.inf]))
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_objective_hessian(of='x', wrt='x', vals=2*np.eye(2))

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_objective_hessian(self, dvs, hess):
        pass

def bounded_convex_lite():
    x0 = np.array([500., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def obj_hess(x):
        return 2 * np.eye(2)
    return ProblemLite(x0, obj=obj, grad=grad, obj_hess=obj_hess, 
                       xl=xl, xu=xu, name='bounded_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_bounded():
    from modopt import CVXOPT
    probs = [BoundedConvex(), bounded_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=6)
        assert_almost_equal(optimizer.results['objective'], 0., decimal=11)

class EqConstrainedConvex(Problem):
    def initialize(self):
        self.problem_name = 'eq_constrained_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]))
        self.add_constraints('c', shape=(1,), equals=np.array([1.]))
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c', wrt='x', vals=np.array([[1.,1.]]))
        self.declare_lagrangian_hessian(of='x', wrt='x', vals=2*np.eye(2))

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_constraints(self, dvs, cons):
        cons['c'][0] = dvs['x'][0] + dvs['x'][1]
    def compute_constraint_jacobian(self, dvs, jac):
        jac['c', 'x'] = np.array([[1.,1.]])
    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        pass

def eq_constrained_convex_lite():
    x0 = np.array([500., 5.])
    cl = np.array([1.])
    cu = np.array([1.])
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def lag_hess(x, lam):
        return 2 * np.eye(2)
    def con(x):
        return np.array([x[0] + x[1]])
    def jac(x):
        return np.array([[1.,1.]])
    return ProblemLite(x0, obj=obj, grad=grad, lag_hess=lag_hess, 
                       con=con, jac=jac, cl=cl, cu=cu,
                       name='eq_constrained_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_eq_constrained():
    from modopt import CVXOPT
    probs = [EqConstrainedConvex(), eq_constrained_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0.5, 0.5], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 0.5, decimal=11)
        assert_array_almost_equal(optimizer.results['constraints'], [1.], decimal=11)

class IneqConstrainedConvex(Problem):
    def initialize(self):
        self.problem_name = 'ineq_constrained_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]))
        self.add_constraints('c', shape=(1,), upper=np.array([1.])) # lower won't be a convex feasible region
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c', wrt='x')
        self.declare_lagrangian_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_constraints(self, dvs, cons):
        cons['c'][0] = dvs['x'][0]**2 - dvs['x'][1]
    def compute_constraint_jacobian(self, dvs, jac):
        jac['c', 'x'] = np.array([[2*dvs['x'][0], -1.]])
    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        lag_hess['x', 'x'] = 2 * np.eye(2) + lag_mult['c'][0] * np.array([[2., 0.], [0., 0.]])

def ineq_constrained_convex_lite():
    x0 = np.array([500., 5.])
    cu = np.array([1.]) # cl won't be a convex feasible region
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def lag_hess(x, lam):
        return 2 * np.eye(2) + lam[0] * np.array([[2., 0.], [0., 0.]])
    def con(x):
        return np.array([x[0]**2 - x[1]])
    def jac(x):
        return np.array([[2*x[0], -1.]])
    return ProblemLite(x0, obj=obj, grad=grad, lag_hess=lag_hess, 
                       con=con, jac=jac, cu=cu, 
                       name='ineq_constrained_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_ineq_constrained():
    from modopt import CVXOPT
    probs = [IneqConstrainedConvex(), ineq_constrained_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0., 0.], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 0., decimal=11)
        assert_array_almost_equal(optimizer.results['constraints'], [0.], decimal=11)

class ConstrainedConvex(Problem):
    def initialize(self):
        self.problem_name = 'constrained_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]))
        self.add_constraints('c', shape=(2,), lower=np.array([1., -np.inf]), upper=np.array([1., 1.]))
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c', wrt='x')
        self.declare_lagrangian_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_constraints(self, dvs, cons):
        cons['c'][0] = dvs['x'][0] + dvs['x'][1]
        cons['c'][1] = dvs['x'][0]**2 - dvs['x'][1]
    def compute_constraint_jacobian(self, dvs, jac):
        jac['c', 'x'] = np.array([[1.,1.], [2*dvs['x'][0], -1.]])
    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        lag_hess['x', 'x'] = 2 * np.eye(2) + lag_mult['c'][0] * np.array([[2., 0.], [0., 0.]])

def constrained_convex_lite():
    x0 = np.array([500., 5.])
    cl = np.array([1., -np.inf])
    cu = np.array([1., 1.])
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def lag_hess(x, lam):
        return 2 * np.eye(2) + lam[0] * np.array([[2., 0.], [0., 0.]])
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1.,1.], [2*x[0], -1.]])
    return ProblemLite(x0, obj=obj, grad=grad, lag_hess=lag_hess, 
                       con=con, jac=jac, cl=cl, cu=cu, 
                       name='constrained_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_constrained():
    from modopt import CVXOPT
    probs = [ConstrainedConvex(), constrained_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0.5, 0.5], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 0.5, decimal=11)
        assert_array_almost_equal(optimizer.results['constraints'], [1., -0.25], decimal=11)

class ConstrainedBoundedConvex(Problem):
    def initialize(self):
        self.problem_name = 'constrained_bounded_convex'
    def setup(self):
        self.add_design_variables('x', shape=(2,), vals=np.array([500., 5.]),
                                  lower=np.array([0., -np.inf]), upper=np.array([np.inf, np.inf]))
        self.add_constraints('c', shape=(2,), lower=np.array([1., -np.inf]), upper=np.array([1., 1.]))
        self.add_objective('f')
    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c', wrt='x')
        self.declare_lagrangian_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**2)
    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']
    def compute_constraints(self, dvs, cons):
        cons['c'][0] = dvs['x'][0] + dvs['x'][1]
        cons['c'][1] = dvs['x'][0]**2 - dvs['x'][1]
    def compute_constraint_jacobian(self, dvs, jac):
        jac['c', 'x'] = np.array([[1.,1.], [2*dvs['x'][0], -1.]])
    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        lag_hess['x', 'x'] = 2 * np.eye(2) + lag_mult['c'][0] * np.array([[2., 0.], [0., 0.]])

def constrained_bounded_convex_lite():
    x0 = np.array([500., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., -np.inf])
    cu = np.array([1., 1.])
    def obj(x):
        return np.sum(x**2)
    def grad(x):    
        return 2 * x
    def lag_hess(x, lam):
        return 2 * np.eye(2) + lam[0] * np.array([[2., 0.], [0., 0.]])
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1.,1.], [2*x[0], -1.]])
    return ProblemLite(x0, obj=obj, grad=grad, lag_hess=lag_hess, 
                       con=con, jac=jac, xl=xl, xu=xu, cl=cl, cu=cu, 
                       name='constrained_bounded_convex_lite')

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_constrained_bounded():
    from modopt import CVXOPT
    probs = [ConstrainedBoundedConvex(), constrained_bounded_convex_lite()]
    solver_options = {'maxiters': 50, 'abstol': 1e-12, 'reltol': 1e-12}

    for prob in probs:
        optimizer = CVXOPT(prob, solver_options=solver_options)
        optimizer.check_first_derivatives(prob.x0)
        optimizer.solve()
        print(optimizer.results)
        optimizer.print_results(optimal_variables= True, 
                                optimal_constraints = True,  
                                optimal_dual_variables = True,  
                                optimal_slack_variables = True)

        assert optimizer.results['status'] == 'optimal'
        assert_array_almost_equal(optimizer.results['x'], [0.5, 0.5], decimal=11)
        assert_almost_equal(optimizer.results['objective'], 0.5, decimal=11)
        assert_array_almost_equal(optimizer.results['constraints'], [1., -0.25], decimal=11)

@pytest.mark.interfaces
@pytest.mark.cvxopt
def test_errors():
    import numpy as np
    from modopt import Problem, ProblemLite, CVXOPT

    # 1. raise error when invalid solver_options are specified
    probs = [UnconstrainedConvex(), unconstrained_convex_lite()]
    for prob in probs:
        with pytest.raises(KeyError) as excinfo:
            optimizer = CVXOPT(prob, solver_options={'verbose':False})
        assert "Option 'verbose' cannot be set because it has not been declared. "\
        "Declared and available options are: ['show_progress', 'maxiters', 'abstol', 'reltol', 'feastol', 'refinement']."\
        in str(excinfo.value)

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


    # 2. Raise error when objective Hessian is not declared for Problem() class
    with pytest.raises(ValueError) as excinfo:
        optimizer = CVXOPT(QPNoHess())
    print(str(excinfo.value))
    assert str(excinfo.value) == "Lagrangian Hessian function is not declared in the Problem() subclass but is needed for CVXOPT."
    
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
    optimizer = CVXOPT(qp_no_hess_lite())
    optimizer.solve()
    optimizer.print_results()
    assert optimizer.results['status'] == 'optimal'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=7)

    class QPNoJac(Problem):
        def setup(self):
            self.add_design_variables('x', shape=(2,))
            self.add_constraints('c', shape=(1,))
            self.add_objective('f')
        def setup_derivatives(self):
            self.declare_objective_gradient(wrt='x', vals=None)
            self.declare_lagrangian_hessian(of='x', wrt='x', vals=2*np.eye(2))

        def compute_objective(self, dvs, obj):
            obj['f'] = np.sum(dvs['x']**2)
        def compute_objective_gradient(self, dvs, grad):
            grad['x'] = 2 * dvs['x']
        def compute_constraints(self, dvs, cons):
            cons['c'][0] = dvs['x'][0] + dvs['x'][1]
        def compute_lagrangian_hessian(self, dvs, lag_mult, hess):
            pass
            
    # 4. Raise error when constraint Jacobian is not declared for Problem() class
    with pytest.raises(ValueError) as excinfo:
        optimizer = CVXOPT(QPNoJac())
    assert str(excinfo.value) == "Constraint Jacobian function is not declared in the Problem() subclass but is needed for CVXOPT."

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
        def lag_hess(x, lam):
            return 2 * np.eye(2)
        def con(x):
            return np.array([x[0] + x[1], x[0] - x[1]])
        
        return ProblemLite(x0, obj=obj, grad=grad, con=con, lag_hess=lag_hess,
                           xl=xl, xu=xu, cl=cl, cu=cu, name='qp_no_jac_lite')

    # 5. Raise warning when constraint Jacobian is not declared for ProblemLite() class and use finite-differences
    optimizer = CVXOPT(qp_no_jac_lite())
    optimizer.solve()
    optimizer.print_results()
    assert optimizer.results['status'] == 'optimal'
    assert_array_almost_equal(optimizer.results['x'], [1., 0.], decimal=6)

if __name__ == '__main__':
    test_unconstrained()
    test_bounded()
    test_eq_constrained()
    test_ineq_constrained()
    test_constrained()
    test_constrained_bounded()
    test_errors()
    print('All tests passed!')