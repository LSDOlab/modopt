__version__ = '0.1.0'

# import modopt base classes
from modopt.core.optimizer import Optimizer
from modopt.core.problem import Problem
from modopt.core.problem_lite import ProblemLite
# from modopt.core.recorder import Recorder
from modopt.core.approximate_hessian import ApproximateHessian
from modopt.core.line_search import LineSearch
from modopt.core.merit_function import MeritFunction
from modopt.core.trust_region import TrustRegion

# import modeling library interfaces
from modopt.external_libraries.csdl import (CSDLProblem, CSDLAlphaProblem)
from modopt.external_libraries.openmdao import (OpenMDAOProblem, )
from modopt.external_libraries.casadi import (CasadiProblem, )
from modopt.external_libraries.jax import (JaxProblem, )

# import test-suite interfaces
from modopt.external_libraries.pycutest import (CUTEstProblem, )

# import external optimizer library interfaces
from modopt.external_libraries.scipy import (SLSQP, COBYLA, BFGS, LBFGSB, TrustConstr, NelderMead)
from modopt.external_libraries.cobyqa import COBYQA
from modopt.external_libraries.pyslsqp import PySLSQP
from modopt.external_libraries.snopt import SNOPTc as SNOPT
from modopt.external_libraries.ipopt import IPOPT
from modopt.external_libraries.qpsolvers import ConvexQPSolvers
from modopt.external_libraries.cvxopt import CVXOPT

# import built-in optimizers
# unconstrained
from modopt.core.optimization_algorithms.steepest_descent import SteepestDescent
from modopt.core.optimization_algorithms.newton import Newton
from modopt.core.optimization_algorithms.quasi_newton import QuasiNewton

# equality-constrained
from modopt.core.optimization_algorithms.newton_lagrange import NewtonLagrange
from modopt.core.optimization_algorithms.l2_penalty_eq import L2PenaltyEq
# from modopt.core.optimization_algorithms.aug_lag import AugLagEq

# general inequality-constrained
# from modopt.core.optimization_algorithms.quadratic_penalty import L2Penalty
# from modopt.core.optimization_algorithms.aug_lag import AugLag

# from modopt.core.optimization_algorithms.interior_point import InteriorPoint

from modopt.core.optimization_algorithms.sqp import SQP as SQPSparse
from modopt.core.optimization_algorithms.sqp_dense import SQP
from modopt.core.optimization_algorithms.sqp_basic import BSQP
from modopt.core.optimization_algorithms.sqp_surf import SQP_SURF

# continuous gradient-free
from modopt.core.optimization_algorithms.nelder_mead_simplex import NelderMeadSimplex
from modopt.core.optimization_algorithms.pso import PSO

# discrete gradient-free
from modopt.core.optimization_algorithms.simulated_annealing import SimulatedAnnealing

# import user-facing functions - need to be imported at the end to avoid circular imports
from modopt.core.optimize import optimize