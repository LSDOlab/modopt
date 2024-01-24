__version__ = '0.1.0'

# import modopt base classes
from modopt.core.optimizer import Optimizer
from modopt.core.problem import Problem
# from modopt.core.recorder import Recorder
from modopt.core.approximate_hessian import ApproximateHessian
from modopt.core.line_search import LineSearch
from modopt.core.merit_function import MeritFunction
from modopt.core.trust_region import TrustRegion

# import csdl library interface
from modopt.external_libraries.csdl.csdl_library import (CSDLProblem, )
# from modopt.external_libraries.openmdao.openmdao_library import (OpenMDAOProblem, )

# import external optimizer library interfaces
from modopt.external_libraries.scipy.scipy_library import (SLSQP, COBYLA, BFGS)
from modopt.external_libraries.snopt.snopt_library import SNOPTc as SNOPT

# import built-in optimizers
# unconstrained
from modopt.core.optimization_algorithms.steepest_descent import SteepestDescent
from modopt.core.optimization_algorithms.newton import Newton
from modopt.core.optimization_algorithms.quasi_newton import QuasiNewton

# equality-constrained
from modopt.core.optimization_algorithms.newton_lagrange import NewtonLagrange
from modopt.core.optimization_algorithms.quadratic_penalty_eq import L2PenaltyEq
# from modopt.core.optimization_algorithms.aug_lag import AugLagEq

# general inequality-constrained
# from modopt.core.optimization_algorithms.quadratic_penalty import L2Penalty
# from modopt.core.optimization_algorithms.aug_lag import AugLag

# from modopt.core.optimization_algorithms.interior_point import InteriorPoint
# from modopt.core.optimization_algorithms.slsqp_v2.slsqp_v2 import SLSQPv2

from modopt.core.optimization_algorithms.sqp import SQP as SQPSparse
from modopt.core.optimization_algorithms.sqp_dense import SQP
from modopt.core.optimization_algorithms.sqp_surf import SQP_SURF

# continuous gradient-free
from modopt.core.optimization_algorithms.nelder_mead import NelderMead
from modopt.core.optimization_algorithms.pso import PSO

# discrete gradient-free
from modopt.core.optimization_algorithms.simulated_annealing import SimulatedAnnealing