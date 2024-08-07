'''Minimizing a Quartic function with constraints and problem scaling'''

import numpy as np
from modopt import Problem

class Quartic(Problem):
    def initialize(self, ):
        self.problem_name = 'quartic'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  scaler=2.,
                                  lower=np.array([0., -np.inf]),
                                  upper=np.array([np.inf, np.inf]),
                                  vals=np.array([50., 5.]))

        self.add_objective('f', scaler=5.0)

        self.add_constraints('c',
                            shape=(2, ),
                            scaler=np.array([10., 100.]),
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
        # jac['c', 'x'] = vals = np.array([[1.,1.],[1.,-1]])

if __name__ == "__main__":

    from modopt import SLSQP, SQP, SNOPT, PySLSQP

    tol = 1E-8
    maxiter = 500

    prob = Quartic(jac_format='dense')
    snopt_options = {
        'Infinite bound': 1.0e20, 
        'Verify level': 3,
        'Major optimality': 1e-8,
        'append2file': True,
        }

    # Set up your optimizer with the problem
    optimizer = SLSQP(prob, solver_options={'maxiter':20}, readable_outputs=['x'])
    # optimizer = PySLSQP(prob, solver_options={'maxiter': 20, 'acc': 1e-6})
    # optimizer = SQP(prob, maxiter=20)
    # optimizer = SNOPT(prob, solver_options=snopt_options)

    optimizer.check_first_derivatives(prob.x0 * prob.x_scaler)
    optimizer.solve()
    optimizer.print_results()

    print('\n')
    print('NOTE: Optimizer and problem Independent Scaling')
    print('===============================================', '\n')
    print('1. Problem() object provides the following unscaled result:')
    print('optimized_dvs:', prob.x.get_data())
    print('optimized_obj:', prob.obj['f'])
    print('optimized_cons:', prob.con.get_data())

    print('\n')
    print('2. Optimizer() object provides the following scaled result:')
    # The following print might not work for interfaced optimizers like SLSQP, COBYLA, SNOPT, PySLSQP ...
    # print('optimize_time:', optimizer.total_time)
    print('optimized_dvs:', optimizer.results['x'])
    # print('optimized_obj:', optimizer.results['objective'])
    # print('optimized_cons:', optimizer.results['constraints'])
