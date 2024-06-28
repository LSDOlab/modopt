
# This file contains the following problem types for both `Problem` and `ProblemLite` classes:
#  1. Unconstrained
#  2. Feasibility
#  3. BoundConstrained
#  4. EqConstrained
#  5. IneqConstrained
#  6. Constrained (both nonlinear and linear eq/ineq constraints with bounds on design variables)
#  7. Scaling (Constrained problem with scaling on design variables, objective, and constraints)
#  8. FiniteDiff (Constrained problem with scaling taht uses finite differencing for grad, hess, hvp, jac, and jvp)
#  9. SecondOrderUnconstrained (Unconstrained problem with scaling that uses analytical objective hessian, for ipopt)
# 10. SecondOrderScaling (Constrained problem with scaling that uses analytical lagrangian hessian, for ipopt)
# 11. ConvexQP (Convex quadratic problem with bounds on design variables and linear eq/ineq constraints for ConvexQPSolvers)


from modopt import Problem, ProblemLite
import numpy as np
class Unconstrained(Problem):
    def initialize(self):
        self.problem_name = 'unconstrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([500., 5.]))
        self.add_objective('f')

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

def unconstrained_lite():
    x0 = np.array([500., 5.])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    
    return ProblemLite(x0, obj=obj, grad=grad, name='unconstrained_lite')

class Feasibility(Problem):
    def initialize(self):
        self.problem_name = 'feasibility'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([500., 5.]))
        self.add_constraints('c',
                             shape=(2, ),
                             lower=np.array([1., 1.]),
                             upper=np.array([1., 1.]),)

    def setup_derivatives(self):
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',)

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0]**2 - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        x = dvs['x']
        jac['c', 'x'] = np.array([[1., 1.], [2.0*x[0], -1]])

def feasibility_lite():
    x0 = np.array([500., 5.])
    cl = np.array([1., 1.])
    cu = np.array([1., 1.])
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1., 1.], [2.0*x[0], -1]])
    
    return ProblemLite(x0, con=con, jac=jac, cl=cl, cu=cu, name='feasibility_lite')

class BoundConstrained(Problem):
    def initialize(self):
        self.problem_name = 'bound_constrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([1., -np.inf]),
                                  upper=np.array([np.inf, 10]),
                                  vals=np.array([50., 5.]))
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        self.add_objective('f')

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

def bound_constrained_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    xl = np.array([1., -np.inf])
    xu = np.array([np.inf, 10])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    
    return ProblemLite(x0, obj=obj, grad=grad, xl=xl, xu=xu, name='bound_constrained_lite')

class EqConstrained(Problem):
    def initialize(self):
        self.problem_name = 'eq_constrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([500., 5.]))
        self.add_objective('f')
        self.add_constraints('c',
                             shape=(1, ),
                             equals=np.array([1.,]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',
                                         vals=np.array([[1., 1.]]))

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['c', 'x'] = vals = np.array([[1., 1.]])

def eq_constrained_lite():
    x0 = np.array([500., 5.])
    ceq = np.array([1.])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def con(x):
        return np.array([x[0] + x[1]])
    def jac(x):
        return np.array([[1., 1.]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, cl=ceq, cu=ceq, name='eq_constrained_lite')

class IneqConstrained(Problem):
    def initialize(self):
        self.problem_name = 'ineq_constrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([50., 5.])) 
                                #   vals=np.array([500., 5.]))  # slsqp diverges when starting from these initial values
        self.add_objective('f')
        self.add_constraints('c',
                             shape=(1, ),
                             lower=np.array([1.]),
                             upper=np.array([np.inf]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',
                                         vals=np.array([[1., -1]]))

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['c', 'x'] = vals = np.array([[1., -1]])

def ineq_constrained_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    cl = np.array([1.])
    cu = np.array([np.inf])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def con(x):
        return np.array([x[0] - x[1]])
    def jac(x):
        return np.array([[1., -1]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, cl=cl, cu=cu, name='ineq_constrained_lite')

class Constrained(Problem):
    # Note: we have a nonlinear constraint here
    def initialize(self):
        self.problem_name = 'constrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  vals=np.array([50., 5.]))
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        self.add_objective('f')
        self.add_constraints('c',
                             shape=(2, ),
                             lower=np.array([1., 1.]),
                             upper=np.array([1., np.inf]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',)

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0]**2 - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        x0 = dvs['x'][0]
        jac['c', 'x'] = vals = np.array([[1., 1.], [2*x0, -1]])

def constrained_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1., 1.], [2*x[0], -1]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, cl=cl, cu=cu, xl=xl, xu=xu, name='constrained_lite')

class Scaling(Problem):
    def initialize(self):
        self.problem_name = 'scaling'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  scaler=np.array([2., 0.2]),
                                  vals=np.array([50., 5.]))
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        
        self.add_objective('f', scaler=20.)
        self.add_constraints('c',
                             shape=(2, ),
                             lower=np.array([1., 1.]),
                             upper=np.array([1., np.inf]),
                             scaler=np.array([5., 0.5]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',)

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0]**2 - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        x0 = dvs['x'][0]
        jac['c', 'x'] = vals = np.array([[1., 1.], [2*x0, -1]])

def scaling_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    x_sc = np.array([2., 0.2])
    o_sc = 20
    c_sc = np.array([5., 0.5])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1., 1.], [2*x[0], -1]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, 
                       cl=cl, cu=cu, xl=xl, xu=xu, 
                       x_scaler=x_sc, o_scaler=o_sc, c_scaler=c_sc, 
                       name='scaling_lite')

class FiniteDiff(Problem):
    def initialize(self):
        self.problem_name = 'finite_diff'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  scaler=np.array([2., 0.2]),
                                  vals=np.array([25., 5.]))
                                #   vals=np.array([50., 5.])) # slsqp diverges when starting from these initial values with FD derivatives
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        self.add_objective('f', scaler=20)
        self.add_constraints('c',
                             shape=(2, ),
                             lower=np.array([1., 1.]),
                             upper=np.array([1., np.inf]),
                             scaler=np.array([5., 0.5]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c', wrt='x',)
        self.declare_objective_hessian(of='x', wrt='x')
        self.declare_objective_hvp(wrt='x')
        self.declare_constraint_jvp(of='c')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        self.use_finite_differencing('objective_gradient', step=1e-6)

    def compute_objective_hessian(self, dvs, hess):
        self.use_finite_differencing('objective_hessian', step=1e-6)

    def compute_objective_hvp(self, dvs, vec, obj_hvp):
        self.use_finite_differencing('objective_hvp', step=1e-6)

    def compute_constraints(self, dvs, con):
        x = dvs['x']
        c= con['c']
        c[0] = x[0] + x[1]
        c[1] = x[0]**2 - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        self.use_finite_differencing('constraint_jacobian', step=1e-6)

    def compute_constraint_jvp(self, dvs, vec, jvp):
        self.use_finite_differencing('constraint_jvp', step=1e-6)


def finite_diff_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    # x0 = np.array([50., 5.])  # slsqp diverges when starting from these initial values when using FD for derivatives
                                # Same problem converges when using analytical derivatives in 'scaling_lite' above
                                # It doesn't converge starting from even x0 = [30., 5.] when using FD for derivatives
                                # However, SNOPT converged even from x0 = [500., 5.]
    x0 = np.array([25., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    x_sc = np.array([2., 0.2])
    o_sc = 20
    c_sc = np.array([5., 0.5])

    def obj(x):
        return np.sum(x**4)
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    
    return ProblemLite(x0, obj=obj, con=con, cl=cl, cu=cu, xl=xl, xu=xu,
                       x_scaler=x_sc, o_scaler=o_sc, c_scaler=c_sc,
                       name='finite_diff_lite')

class SecondOrderUnconstrained(Problem):
    def initialize(self):
        self.problem_name = 'second_order_unconstrained'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  scaler=np.array([2., 0.2]),
                                  vals=np.array([50., 5.]))
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        
        self.add_objective('f', scaler=20.)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_objective_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_objective_hessian(self, dvs, obj_hess):
        x = dvs['x']
        obj_hess['x', 'x'] = np.array([[12*x[0]**2, 0], [0, 12*x[1]**2]])

def second_order_unconstrained_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    x_sc = np.array([2., 0.2])
    o_sc = 20
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def obj_hess(x):
        return np.array([[12*x[0]**2, 0], [0, 12*x[1]**2]])
    
    return ProblemLite(x0, obj=obj, grad=grad,
                       xl=xl, xu=xu, 
                       x_scaler=x_sc, o_scaler=o_sc,
                       obj_hess=obj_hess, name='second_order_unconstrained_lite')

class SecondOrderScaling(Problem):
    def initialize(self):
        self.problem_name = 'second_order_scaling'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  scaler=np.array([2., 0.2]),
                                  vals=np.array([50., 5.]))
                                #   vals=np.array([500., 5.])) # slsqp diverges when starting from these initial values
        
        self.add_objective('f', scaler=20.)
        self.add_constraints('c',
                             shape=(2, ),
                             lower=np.array([1., 1.]),
                             upper=np.array([1., np.inf]),
                             scaler=np.array([5., 0.5]),)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x')
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',)
        self.declare_lagrangian_hessian(of='x', wrt='x')

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0]**2 - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        x0 = dvs['x'][0]
        jac['c', 'x'] = vals = np.array([[1., 1.], [2*x0, -1]])

    def compute_lagrangian_hessian(self, dvs, lag_mult, lag_hess):
        x = dvs['x']
        l = lag_mult['c']
        lag_hess['x', 'x'] = np.array([[12*x[0]**2, 0], [0, 12*x[1]**2]]) + l[0] * np.array([[0, 0], [0, 0]]) + l[1] * np.array([[2, 0], [0, 0]])

def second_order_scaling_lite():
    # x0 = np.array([500., 5.]) # slsqp diverges when starting from these initial values
    x0 = np.array([50., 5.])
    xl = np.array([0., -np.inf])
    xu = np.array([np.inf, np.inf])
    cl = np.array([1., 1.])
    cu = np.array([1., np.inf])
    x_sc = np.array([2., 0.2])
    o_sc = 20
    c_sc = np.array([5., 0.5])
    def obj(x):
        return np.sum(x**4)
    def grad(x):    
        return 4 * x ** 3
    def con(x):
        return np.array([x[0] + x[1], x[0]**2 - x[1]])
    def jac(x):
        return np.array([[1., 1.], [2*x[0], -1]])
    def lag_hess(x, l):
        return np.array([[12*x[0]**2, 0], [0, 12*x[1]**2]]) + l[0] * np.array([[0, 0], [0, 0]]) + l[1] * np.array([[2, 0], [0, 0]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, 
                       cl=cl, cu=cu, xl=xl, xu=xu, 
                       x_scaler=x_sc, o_scaler=o_sc, c_scaler=c_sc, 
                       lag_hess=lag_hess, name='second_order_scaling_lite')

class ConvexQP(Problem):
    def initialize(self, ):
        self.problem_name = 'convex_qp'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  vals=np.array([500., 5.]))

        self.add_objective('f')

        self.add_constraints('c',
                            shape=(2, ),
                            lower=np.array([1., 1.]),
                            upper=np.array([1., np.inf]),
                            equals=None,)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x', vals=None)
        self.declare_objective_hessian(of='x', wrt='x', vals=2*np.eye(2))
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',
                                         vals=np.array([[1.,1.],[1.,-1]]))

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**2)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 2 * dvs['x']

    def compute_objective_hessian(self, dvs, hess):
        pass

    def compute_constraints(self, dvs, cons):
        x   = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0] - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        pass

def convex_qp_lite():
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
    def jac(x):
        return np.array([[1., 1.], [1., -1]])
    
    return ProblemLite(x0, obj=obj, grad=grad, con=con, jac=jac, obj_hess=obj_hess,
                       xl=xl, xu=xu, cl=cl, cu=cu, name='convex_qp_lite')