import numpy as np

from rosenbrock import Rosenbrock as Ros

from modopt.optimization_algorithms import SteepestDescent, Newton, QuasiNewton

tol = 1E-8
max_itr = 500

prob = Ros()

optimizer = SteepestDescent(prob, opt_tol=tol, max_itr=max_itr)
optimizer.setup()
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(opt_summary=True)

# prob = Ros(nx=nx, x0=x0)

# optimizer = QuasiNewton(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.setup()
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(opt_summary=True)

# prob = Ros(nx=nx, x0=x0)

# optimizer = Newton(prob, opt_tol=tol, max_itr=max_itr)
# optimizer.setup()
# optimizer.check_first_derivatives(prob.x.get_data())
# optimizer.solve()
# optimizer.print_results(opt_summary=True)
# optimizer.print_results(opt_summary=True, compact_print=True)
