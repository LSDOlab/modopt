import numpy as np

from rosenbrock_sqp import Rosenbrock as R

from modopt.optimization_algorithms import SQP
from modopt.scipy_library import SLSQP

opt_tol = 1E-8
feas_tol = 1E-8
max_itr = 500

prob = R()

optimizer = SLSQP(prob, opt_tol=opt_tol, feas_tol=feas_tol)
optimizer.solve()
optimizer.print_results(summary_table=True)
