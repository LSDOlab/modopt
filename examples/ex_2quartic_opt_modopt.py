import numpy as np
from modopt.api import Problem

class Quartic(Problem):
    def initialize(self, ):
        self.problem_name = 'quartic'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(1, ),
                                  lower=np.array([0.,]),
                                  upper=None,
                                  vals=np.array([500.,]))

        self.add_design_variables('y',
                                  shape=(1, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.array([5.,]))

        self.add_objective('f')

        self.add_constraints('x+y',
                            shape=(1, ),
                            lower=None,
                            upper=None,
                            equals=np.array([1.,]),)

        self.add_constraints('x-y',
                            shape=(1, ),
                            lower=np.array([1.,]),
                            upper=None,
                            equals=None,)

    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x', vals=None)
        self.declare_objective_gradient(wrt='y', vals=None)
        self.declare_constraint_jacobian(of='x+y',
                                         wrt='x',
                                         vals=np.array([1.,]))
        self.declare_constraint_jacobian(of='x+y',
                                         wrt='y',
                                         vals=np.array([1.,]))
        self.declare_constraint_jacobian(of='x-y',
                                         wrt='x',
                                         vals=np.array([1.,]))
        self.declare_constraint_jacobian(of='x-y',
                                         wrt='y',
                                         vals=np.array([-1.,]))

    def compute_objective(self, dvs, obj):
        obj['f'] = dvs['x']**4 + dvs['y']**4

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x'] ** 3
        grad['y'] = 4 * dvs['y'] ** 3

    def compute_constraints(self, dvs, cons):
        cons['x+y'] = dvs['x'] + dvs['y']
        cons['x-y'] = dvs['x'] - dvs['y']

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # jac['x+y', 'x'] = 1.
        # jac['x+y', 'y'] = 1.
        # jac['x-y', 'x'] = 1.
        # jac['x-y', 'y'] = -1.


from modopt.scipy_library import SLSQP
# from modopt.optimization_algorithms import SQP
from modopt.snopt_library import SNOPT

tol = 1E-8
max_itr = 500

prob = Quartic(jac_format='dense')

# Set up your optimizer with the problem
optimizer = SLSQP(prob, maxiter=20)
# optimizer = SQP(prob, max_itr=20)
# optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)

optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)

print('optimized_dvs:', prob.x.get_data())
print('optimized_cons:', prob.con.get_data())
print('optimized_obj:', prob.obj['f'])
