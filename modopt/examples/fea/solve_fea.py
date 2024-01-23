import numpy as np

from fea import FEA

from modopt.scipy_library import SLSQP, COBYLA

ftol = 1E-7
tol = ftol
maxiter = 500

nx = 40
num_elements = 40
forces = np.ones(num_elements)
prob = FEA(nx=nx, num_elements=num_elements, forces=forces)

optimizer = SLSQP(prob, ftol=ftol, maxiter=maxiter)
# optimizer = COBYLA(prob, tol=tol, maxiter=maxiter)

optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
# optimizer.print_results()
optimizer.print_results(optimal_variables=True)