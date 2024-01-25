'''Example 6 : Quartic optimization using Jax'''

import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)

import numpy as np
from modopt import Problem

# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

def objective(x):
    return jnp.sum(x ** 4)

def constraints(x):
    return jnp.array([x[0] + x[1], x[0] - x[1]])

# def con2(x, y):
#     return x - y

obj_func = jax.jit(objective)
grad_func = jax.jit(jax.grad(objective))

con_func = jax.jit(constraints)
jac_func = jax.jit(jax.jacfwd(constraints))

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
        jac_0 = np.array(jac_func(np.array([500., 5.])))
        self.declare_constraint_jacobian(of='c',
                                         wrt='x',
                                        vals=jac_0)

    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = np.float_(obj_func(x))

    def compute_objective_gradient(self, dvs, grad):
        x = dvs['x']
        grad['x'] = np.array(grad_func(x))

    def compute_constraints(self, dvs, cons):
        x = dvs['x']
        cons['c'] = np.array(con_func(x))

    def compute_constraint_jacobian(self, dvs, jac):
        pass
        # x = dvs['x']
        # jac['c', 'x'] = np.array(jac_func(x))


if __name__ == "__main__":
    # Instantiate your problem using the csdl Simulator object and name your problem
    prob = Quartic(jac_format='dense')

    from modopt import SLSQP, SQP, SNOPT, COBYLA

    # Setup your preferred optimizer (here, SLSQP) with the Problem object 
    # Pass in the options for your chosen optimizer
    optimizer = SLSQP(prob, maxiter=20)
    # optimizer = SQP(prob, max_itr=20)
    # optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3, Verbose=True)

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization (summary_table contains information from each iteration)
    optimizer.print_results(summary_table=True)

    print('optimized_dvs:', prob.x.get_data())
    print('optimized_cons:', prob.con.get_data())
    print('optimized_obj:', prob.obj['f'])

