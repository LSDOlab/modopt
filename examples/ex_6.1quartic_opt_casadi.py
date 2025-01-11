'''Quartic optimization using CasADi'''

import numpy as np
import modopt as mo
import casadi as ca

# minimize x^4 + y^4 subject to x>=0, x+y=1, x-y>=1.

# METHOD 1: Use CasADi expressions directly in mo.CasadiProblem.
#           modOpt will auto-generate the gradient, Jacobian, and objective Hessian.
#           modOpt will also auto-generate the Lagrangian, its gradient, and Hessian.
#           No need to manually generate functions or their derivatives and then wrap them.

ca_obj = lambda x: ca.sum1(x**4)
ca_con = lambda x: ca.vertcat(x[0] + x[1], x[0] - x[1])

prob = mo.CasadiProblem(x0=np.array([500., 5.]), ca_obj=ca_obj, ca_con=ca_con, 
                        cl=np.array([1., 1.]), cu=np.array([1., np.inf]), 
                        xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
                        name='quartic_casadi', order=1)

# # METHOD 2: Create CasADi functions and derivatives, 
# #           and wrap them manually before passing to Problem/ProblemLite.

# # Create scalar/matrix symbols
# x = ca.MX.sym('x', 2)

# # Compose into expressions to evaluate objective
# obj_expr = ca.sum1(x**4)

# # Sensitivity of expression -> new expression
# grad_expr = ca.gradient(obj_expr,x)

# # Create a Function to evaluate obj and grad expressions
# _obj  = ca.Function('o',[x],[obj_expr])
# _grad = ca.Function('g',[x],[grad_expr])

# # Create an expression for the constraints and their jacobian
# con_expr = ca.vertcat(x[0] + x[1], x[0] - x[1])
# jac_expr = ca.jacobian(con_expr, x)

# # Create a Function to evaluate con and jac expressions
# _con = ca.Function('c', [x], [con_expr])
# _jac = ca.Function('j', [x], [jac_expr])

# # Wrap the functions for modopt
# obj  = lambda x: np.float64(_obj(x))
# grad = lambda x: np.array(_grad(x)).flatten()
# con  = lambda x: np.array(_con(x)).flatten()
# jac  = lambda x: np.array(_jac(x))

# # Create a ProblemLite object using the wrapped functions
# prob = mo.ProblemLite(x0=np.array([500., 5.]), 
#                       obj=obj, grad=grad, con=con, jac=jac,
#                       cl=np.array([1., 1.]), cu=np.array([1., np.inf]),
#                       xl=np.array([0., -np.inf]), xu=np.array([np.inf, np.inf]),
#                       name = 'quartic_casadi')

if __name__ == "__main__":

    import modopt as mo

    # Setup your preferred optimizer (here, SLSQP) with the Problem object 
    # Pass in the options for your chosen optimizer
    optimizer = mo.SLSQP(prob, solver_options={'maxiter':20})

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization
    optimizer.print_results()

    print('optimized_dvs:', optimizer.results['x'])
    print('optimized_obj:', optimizer.results['fun'])