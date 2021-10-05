import numpy as np

from x4 import X4

from modopt.optimization_algorithms import SteepestDescent, Newton, BFGS
from modopt.scipy_library import SLSQP

# nx = 25
tol = 1E-8
max_itr = 500
# x0 = np.full((nx), 0.1)

# prob = X4(nx=nx, x0=x0)
prob = X4()

optimizer = SteepestDescent(prob, opt_tol=tol, max_itr=max_itr)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)

prob = X4()

optimizer = Newton(prob, opt_tol=tol, max_itr=max_itr)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)

prob = X4()

optimizer = BFGS(prob, opt_tol=tol, max_itr=max_itr)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)
# optimizer.print_results(summary_table=True, compact_print=True)

# tol = 1E-12

# optimizer = SLSQP(prob, ftol=tol, max_itr=max_itr)
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(optimal_variables=True)