import numpy as np

from fea import FEA

from modopt.scipy_library import SLSQP, COBYLA

# nx = 25
tol = 1E-8
max_itr = 500
# x0 = np.full((nx), 0.1)

# prob = X4(nx=nx, x0=x0)
nx = 10
num_elements = 40
# forces = np.ones(num_elements + 1)
forces = np.ones(num_elements)
prob = FEA(nx=nx, num_elements=num_elements, forces=forces)

optimizer = SLSQP(prob, opt_tol=tol, max_itr=max_itr)
# optimizer = COBYLA(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(opt_summary=True)

# optimizer.print_results(opt_summary=True, compact_print=True)