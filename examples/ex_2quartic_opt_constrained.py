'''Minimizing a Quartic function with constraints'''

import numpy as np
from modopt import Problem

class Quartic(Problem):
    def initialize(self, ):
        self.problem_name = 'quartic'

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
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',
                                        vals=np.array([[1.,1.],[1.,-1]]))

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.sum(x**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3

    def compute_constraints(self, dvs, cons):
        x   = dvs['x']
        con = cons['c']
        con[0] = x[0] + x[1]
        con[1] = x[0] - x[1]

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['c', 'x'] = vals=np.array([[1.,1.],[1.,-1]])

from modopt import SLSQP, SQP, SNOPT

tol = 1E-8
maxiter = 500

prob = Quartic(jac_format='dense')

print(prob)

snopt_options = {
    'Infinite bound': 1.0e20, 
    'Verify level': 3,
    }

# Set up your optimizer with the problem
optimizer = SLSQP(prob, solver_options={'maxiter':20})
# optimizer = SQP(prob, maxiter=20)
# optimizer = SNOPT(prob, solver_options=snopt_options)

optimizer.check_first_derivatives(prob.x0)
optimizer.solve()
optimizer.print_results(summary_table=True)

print(prob)

print('optimized_dvs:', prob.x.get_data())
print('optimized_cons:', prob.con.get_data())
print('optimized_obj:', prob.obj['f'])
