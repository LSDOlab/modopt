import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from rosenbrock_2d_sqp import Rosenbrock2d as R2

import sys

sys.path.append("..")

from modopt.api import SteepestDescent, Newton, QuasiNewton
from modopt.api import SQP_OSQP

nx = 2
tol = 1E-8
x0 = np.array([0., 50])
# x0 = np.array([0, -np.sqrt(2)])
# x0 = np.array([np.sqrt(2), 0])

prob = R2(nx=nx, nc=nx)

opt = SQP_OSQP(prob, x0=x0, opt_tol=1e-8, feas_tol=1e-8)
opt.setup()
opt.run()
opt.print(table=True)
