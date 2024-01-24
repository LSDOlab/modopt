import numpy as np

from x4_sqp import X4

from modopt import SQP
from modopt import SLSQP

opt_tol = 1E-8
feas_tol = 1E-8
max_itr = 50

prob = X4()

optimizer = SQP(prob,
                opt_tol=opt_tol,
                feas_tol=feas_tol,
                max_itr=max_itr)
# optimizer = SLSQP(prob,
#                   ftol=opt_tol,
#                   opt_tol=opt_tol,
#                   feas_tol=feas_tol,
#                   max_itr=max_itr)
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True,
                        optimal_variables=True,
                        constraints=True)
