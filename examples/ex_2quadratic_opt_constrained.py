'''Minimizing a Quartic function with constraints'''

import numpy as np
from modopt import Problem

class Quadratic(Problem):
    def initialize(self, ):
        self.problem_name = 'quadratic'

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
        self.declare_lagrangian_hessian(of='x', wrt='x', vals=2*np.eye(2))
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

    def compute_lagrangian_hessian(self, dvs, lag_mult, hess):
        pass

    def compute_constraints(self, dvs, cons):
        x   = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0] - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['c', 'x'] = vals=np.array([[1.,1.],[1.,-1]])

from modopt import SLSQP, SQP, SNOPT , ConvexQPSolvers, CVXOPT

tol = 1E-8
maxiter = 500

prob = Quadratic(jac_format='dense')

print(prob)

# Set up your optimizer with the problem
# solver_options = {'solver':'quadprog'}
# optimizer = ConvexQPSolvers(prob, solver_options=solver_options)
# optimizer.check_first_derivatives(prob.x0)
# optimizer.solve()
# optimizer.print_results(optimal_variables=True,
#                         optimal_constraints=True,
#                         optimal_dual_variables=True,
#                         extras=True)

# optimizer = SLSQP(prob, maxiter=20)
# optimizer = SQP(prob, maxiter=20)
# optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)

solver_options = {'maxiters':10, 'abstol':1e-7, 'reltol':1e-6, 'feastol':1e-7}
optimizer = CVXOPT(prob, solver_options=solver_options)
optimizer.solve()
optimizer.print_results(optimal_variables=True,
                        optimal_constraints=True,
                        optimal_dual_variables=True,
                        optimal_slack_variables=True)

# print('optimized_dvs:', prob.x.get_data())
# print('optimized_cons:', prob.con.get_data())
# print('optimized_obj:', prob.obj['f'])

print(prob)
