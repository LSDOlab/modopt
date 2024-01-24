import numpy as np

from x4_sqp import X4

from modopt import SteepestDescent, Newton, QuasiNewton
from modopt import SQP_OSQP

nx = 1000
tol = 1E-8
x0 = np.full((nx), 5.)

prob = X4(nx=nx, nc=nx)

optimizer = SQP_OSQP(prob, x0=x0, opt_tol=1e-8, feas_tol=1e-8)
optimizer.setup()
optimizer.run()
optimizer.print(table=True)
