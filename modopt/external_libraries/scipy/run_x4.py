import numpy as np

from x4 import X4

from modopt.scipy_library import SLSQP, COBYLA, BFGS

nx = 25
tol = 1E-8
max_itr = 500
x0 = np.full((nx), 0.1)

prob = X4(nx=nx, x0=x0)

optimizer = COBYLA(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.check_first_derivatives(x0)
optimizer.solve()
# optimizer.print_results(opt_summary=True)