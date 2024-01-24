import numpy as np

from quadratic import Quadratic

from modopt import SLSQP
from modopt import SQP
from modopt import SNOPT

# nx = 25
tol = 1E-8
max_itr = 500
# x0 = np.full((nx), 0.1)

# prob = X4(nx=nx, x0=x0)
prob = Quadratic(jac_format='dense')

# Setup your optimizer with the problem
# optimizer = SLSQP(prob, maxiter=20)
# optimizer = SQP(prob, max_itr=20)
optimizer = SNOPT(prob, Infinite_bound=1.0e20, Verify_level=3)

optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
# optimizer.print_results(summary_table=True)
# print(optimizer.outputs['x'][-1])

print('optimized_dvs:', prob.x.get_data())
print('optimized_cons:', prob.con.get_data())

# print('dvs:', optimizer.snopt_output.x[:len(prob.x.get_data())])
# print('cons:', optimizer.snopt_output.x[len(prob.x.get_data()):])