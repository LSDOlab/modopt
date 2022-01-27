import numpy as np

from x4 import X4

from modopt.optimization_algorithms import SteepestDescent, Newton, QuasiNewton
from modopt.scipy_library import SLSQP

tol = 1E-8
max_itr = 500

prob = X4()

# optimizer = SteepestDescent(prob,
#                             opt_tol=tol,
#                             max_itr=max_itr,
#                             outputs=[
#                                 'itr',
#                                 'obj',
#                                 'x',
#                                 'opt',
#                                 'time',
#                             ])
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(summary_table=True)

# prob = X4()

# optimizer = Newton(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(summary_table=True)

# prob = X4()

# optimizer = QuasiNewton(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(summary_table=True)
# # optimizer.print_results(summary_table=True, compact_print=True)

ftol = 1E-12
maxiter = 500

optimizer = SLSQP(prob, ftol=ftol, maxiter=maxiter)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results()