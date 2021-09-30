import numpy as np

from quadratic import Quadratic

from modopt.scipy_library import SLSQP

# nx = 25
tol = 1E-8
max_itr = 500
# x0 = np.full((nx), 0.1)

# prob = X4(nx=nx, x0=x0)
prob = Quadratic()

optimizer = SLSQP(prob, opt_tol=tol, max_itr=max_itr)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(opt_summary=True)
