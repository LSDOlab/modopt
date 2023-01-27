
from simple_example import X4

from modopt.optimization_algorithms import SteepestDescent, Newton, QuasiNewton

tol = 1E-8
max_itr = 500

prob = X4()

optimizer = QuasiNewton(
    prob,
    opt_tol=tol,
    max_itr=max_itr,
    )

optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)
