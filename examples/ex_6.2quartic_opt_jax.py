'''Quartic optimization using Jax'''

import numpy as np
import modopt as mo

import jax
import jax.numpy as jnp 
jax.config.update("jax_enable_x64", True)

# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

jax_obj = lambda x: jnp.sum(x ** 4)
jax_con = lambda x: jnp.array([x[0] + x[1], x[0] - x[1]])

# METHOD 1: Use Jax functions directly in mo.JaxProblem. 
#           modOpt will auto-generate gradient, Jacobian, and objective Hessian.
#           modOpt will also auto-generate the Lagrangian, its gradient, and Hessian.
#           No need to manually generate or jit functions or their derivatives and then wrap them.

prob = mo.JaxProblem(x0=np.array([500., 5.]), nc=2, jax_obj=jax_obj, jax_con=jax_con,
                     xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
                     cl=np.array([1., 1.]), cu=np.array([1., np.inf]), 
                     name='quartic', order=1)

# # METHOD 2: Create jitted Jax functions and derivatives, and
# #           wrap them manually before passing to Problem/ProblemLite.

# jax_obj = lambda x: jnp.sum(x ** 4)
# jax_con = lambda x: jnp.array([x[0] + x[1], x[0] - x[1]])

# _obj  = jax.jit(jax_obj)
# _grad = jax.jit(jax.grad(jax_obj))
# _con  = jax.jit(jax_con)
# _jac  = jax.jit(jax.jacfwd(jax_con))

# obj  = lambda x: np.float64(_obj(x))
# grad = lambda x: np.array(_grad(x))
# con  = lambda x: np.array(_con(x))
# jac  = lambda x: np.array(_jac(x))

# prob = mo.ProblemLite(x0=np.array([500., 5.]), obj=obj, grad=grad, con=con, jac=jac,
#                       xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
#                       name='quartic', cl=np.array([1., 1.]), cu=np.array([1., np.inf]))

# class Quartic(mo.Problem):
#     def initialize(self, ):
#         self.problem_name = 'quartic'

#     def setup(self):
#         self.add_design_variables('x',
#                                   shape=(2, ),
#                                   lower=np.array([0., -np.inf]),
#                                   upper=np.array([np.inf, np.inf]),
#                                   vals=np.array([500., 5.]))

#         self.add_objective('f')

#         self.add_constraints('c',
#                             shape=(2, ),
#                             lower=np.array([1., 1.]),
#                             upper=np.array([1., np.inf]),
#                             equals=None,)

#     def setup_derivatives(self):
#         self.declare_objective_gradient(wrt='x', vals=None)
#         jac_0 = jac(np.array([500., 5.]))
#         self.declare_constraint_jacobian(of='c',
#                                          wrt='x',
#                                          vals=jac_0)

#     def compute_objective(self, dvs, o):
#         x = dvs['x']
#         o['f'] = obj(x)

#     def compute_objective_gradient(self, dvs, g):
#         x = dvs['x']
#         g['x'] = grad(x)

#     def compute_constraints(self, dvs, c):
#         x = dvs['x']
#         c['c'] = con(x)

#     def compute_constraint_jacobian(self, dvs, j):
#         pass
#         # x = dvs['x']
#         # j['c', 'x'] = jac(x)

# prob = Quartic(jac_format='dense')

if __name__ == "__main__":

    import modopt as mo

    # Setup your preferred optimizer (here, SLSQP) with the Problem object 
    # Pass in the options for your chosen optimizer
    optimizer = mo.SLSQP(prob, solver_options={'maxiter':20})

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization (summary_table contains information from each iteration)
    optimizer.print_results()

    print('optimized_dvs:', optimizer.results['x'])
    print('optimized_obj:', optimizer.results['fun'])