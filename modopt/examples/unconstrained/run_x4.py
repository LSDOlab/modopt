import numpy as np

from x4 import X4

from modopt.optimization_algorithms import SteepestDescent, Newton, QuasiNewton
from modopt.scipy_library import SLSQP

tol = 1E-8
max_itr = 500

prob = X4()

optimizer = SteepestDescent(
    prob,
    opt_tol=tol,
    max_itr=100,
    outputs=['itr', 'obj', 'x', 'opt', 'time', 'step'])
# outputs=['itr', 'obj', 'x', 'opt', 'time', 'step', 'test'])
optimizer.check_first_derivatives(prob.x0)
optimizer.solve()
optimizer.print_results(summary_table=True, compact_print=True)

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

# optimizer = SLSQP(prob, ftol=ftol, maxiter=maxiter)
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results()