import numpy as np

from rosenbrock_sqp import Rosenbrock as R

from modopt import SQP
from modopt import SLSQP

opt_tol = 1E-8
feas_tol = 1E-8
max_itr = 500

prob = R()

optimizer = SLSQP(prob, opt_tol=opt_tol, feas_tol=feas_tol)
optimizer.solve()
optimizer.print_results(summary_table=True)
