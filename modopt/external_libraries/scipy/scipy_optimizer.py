import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS, SR1
from array_manager.api import DenseMatrix

from modopt import Optimizer, Problem

import warnings

# scipy.optimize.minimize(fun,
#                         x0,
#                         args=(),
#
#                         method=None, {str or callable}, optional
#                         # If not given, chosen to be one of BFGS, L-BFGS-B, SLSQP,
#                         # depending on if the problem has constraints or bounds.
#
#                         jac=None, {callable, ‘2-point’, ‘3-point’, ‘cs’, bool}, optional
#                         # method returning gradient vector
#                         # Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg,
#                         # trust-ncg, trust-krylov, trust-exact and trust-constr.
#                         # If jac is a Boolean and is True, fun is assumed to return
#                         # a tuple (f, g) containing the objective function and the gradient.
#                         # If None or False, the gradient will be estimated using 2-point
#                         # finite difference estimation with an absolute step size.
#                         # Alternatively, the keywords {‘2-point’, ‘3-point’, ‘cs’} can be used.
#
#                         hess=None, {callable, ‘2-point’, ‘3-point’, ‘cs’, HessianUpdateStrategy}, optional
#                         # method returning Hessian matrix
#                         # Only for Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact
#                         # and trust-constr
#
#                         hessp=None, callable, optional
#                         # Hessian of objective function times an arbitrary vector p.
#                         # Only for Newton-CG, trust-ncg, trust-krylov, trust-constr.
#                         # Only one of hessp(x,p) or hess(x) needs to be given.
#                         # If hess is provided, then hessp will be ignored.
#
#                         bounds=None,
#                         # Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
#                         # Powell, trust-constr, COBYLA, and COBYQA methods.
#                         # There are two ways to specify the bounds:
#                         # Instance of Bounds() class.
#                         # Sequence of (min, max) pairs for each element in x.
#                         # None is used to specify no bound.
#
#                         constraints=(),
#                         # Constraints definition (only for COBYLA, COBYQA, SLSQP and trust-constr).
#                         # ‘trust-constr’ and 'COBYQA': single object (LinearConstraint or NonlinearConstraint) or
#                         # a list of objects specifying constraints to the optimization problem.
#                         # 'COBYLA', 'SLSQP': Constraints are defined as a list of dictionaries.
#                         # {'type' : 'eq' or 'ineq', 
#                         #  'fun' : callable constraint function,
#                         #  'jac' : callable Jacobian of fun (optional),
#                         #  'args' : Extra arguments to be passed to fun and jac.}
#                         # Equality constraint means c(x) = 0
#                         # whereas inequality means c(x) >= 0
#                         # Note that COBYLA only supports inequality constraints.
#
#                         tol=None,
#                         # solver-specific tolerance(s) for termination.
#                         # For detailed control, use solver-specific options.
#
#                         callback=None,
#                         # Called after each iteration. For ‘trust-constr’, callable with the signature:
#                         # callback(xk, OptimizeResult state) -> bool
#                         # where xk is the current d.v. and state is an OptimizeResult() object, with the
#                         # same fields as the ones from the return.
#                         # If callback returns True, the algorithm execution is terminated.
#                         # For all the other methods, the signature is:
#                         # callback(xk)
#                         # where xk is the current parameter vector.
#
#                         options=None)
#                         # A dictionary of solver options.
#                         # All methods except TNC accept the following generic options:
#                         # options = {maxiter:int, disp:bool (True to print convergence messages),
#                         # +solver_specific_options}, given by
#                         # scipy.optimize.show_options(solver=None, method=None, disp=True)

# Returns : OptimizeResult() object
# Important attributes are: x the solution array,
# success a Boolean flag indicating if the optimizer exited successfully and
# message which describes the cause of the termination.
# See OptimizeResult for a description of other attributes (x, success, status, messsage, fun, jac, hess,
# hess_inv, nfev, njev, nhev, nit, maxcv).
# There may be additional attributes not listed above depending of the specific solver.
# View available atributes using keys() method of dict eg. result_obj.keys() .


class ScipyOptimizer(Optimizer):
    def initialize(self):
        self.solver_name = 'scipy_'
        self.options.declare('gradient',
                             default='2-point',
                             values=['2-point', '3-point', 'cs'])
        self.options.declare('jacobian',
                             default='2-point',
                             values=['2-point', '3-point', 'cs'])
        self.options.declare(
            'hessian',
            default='2-point',
            values=['2-point', '3-point', 'cs', 'bfgs', 'sr1'])

        self.options.declare(
            'lagrangian_hessian',
            default='2-point',
            values=['2-point', '3-point', 'cs', 'bfgs', 'sr1'])

        # Declare method specific options (implemented in the respective algorithm)
        self.declare_options()
        self.declare_outputs()

        self.obj = self.problem._compute_objective
        self.x0 = self.problem.x0

        # Gradient only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg,
        # trust-krylov, trust-exact and trust-constr
        # TODO: A recent change, look into this and see if there are any issues, test this for auto-FD capability
        # if not isinstance(self.problem, CSDLProblem):
        if self.problem.compute_objective_gradient.__func__ is not Problem.compute_objective_gradient:
            self.grad = self.problem._compute_objective_gradient
        else:
            self.grad = self.options['gradient']
        # else:     
        #     self.grad = self.problem._compute_objective_gradient

        # Only for COBYLA, SLSQP and trust-constr
        # Used to construct:
        # 1. a single object or a list of constraint() objects for trust-constr
        # 2. a list of dictionaries for COBYLA and SLSQP

        if self.problem.nc > 0:
            # Uncomment the line below after testing our sqp_optimizer with atomics_lite
            # pC_px = DenseMatrix(self.problem.pC_px).numpy_array()
            self.con = self.problem._compute_constraints
        # TODO: A recent change, look into this and see if there are any issues, test this for auto-FD capability
            # if not isinstance(self.problem, CSDLProblem):
            if self.problem.compute_constraint_jacobian.__func__ is not Problem.compute_constraint_jacobian:
                self.jac = self.problem._compute_constraint_jacobian

            # Uncomment the 2 lines below after testing our sqp_optimizer with atomics_lite
            # elif pC_px.any() != 0:
            #     self.jac = lambda x: pC_px
            else:
                warnings.warn("Empty compute_constraint_jacobian() method for Problem() needs to be defined even if your "+ 
                              "constraint Jacobians are constants.")
                self.jac = self.options['jacobian']
            # else:
            #     self.jac = self.problem._compute_constraint_jacobian

        # SETUP OBJECTIVE HESSIAN OR HVP

        if self.problem.compute_objective_hessian.__func__ is not Problem.compute_objective_hessian:
            self.hess = self.problem._compute_objective_hessian
        elif self.options['hessian'] == 'BFGS':
            # Users will have to manually modify this if needed
            self.hess = BFGS(exception_strategy='skip_update',
                             min_curvature=None,
                             init_scale='auto')
        elif self.options['hessian'] == 'SR1':
            # Users will have to manually modify this if needed
            self.hess = SR1(min_denominator=1e-08, init_scale='auto')

        # Only for Newton-CG, trust-ncg, trust-krylov, trust-constr.
        # Note: This is always Objective hessian even for trust-constr (Constraint hessians defined in constraints)
        elif self.problem.compute_objective_hvp.__func__ is not Problem.compute_objective_hvp:
            self.hvp = self.problem._compute_objective_hvp
        else:
            self.hess = self.options['hessian']

        # SETUP LAGRANGIAN HESSIAN (only for trust-constr)
        if self.problem.compute_lagrangian_hessian.__func__ is not Problem.compute_lagrangian_hessian:
            self.lag_hess = self.problem.compute_lagrangian_hessian

        elif self.options['lagrangian_hessian'] == 'BFGS':
            # Users will have to manually modify this if needed
            self.lag_hess = BFGS(exception_strategy='skip_update',
                                 min_curvature=None,
                                 init_scale='auto')
        elif self.options['lagrangian_hessian'] == 'SR1':
            # Users will have to manually modify this if needed
            self.lag_hess = SR1(min_denominator=1e-08,
                                init_scale='auto')

        else:
            self.lag_hess = self.options['lagrangian_hessian']

    def setup_bounds(self):
        # Adapt bounds as scipy Bounds() object
        # Only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, COBYLA, and COBYQA methods
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        if np.all(xl == -np.inf) and np.all(xu == np.inf):
            self.bounds = None
        else:
            self.bounds = Bounds(xl, xu, keep_feasible=False)

    def setup_constraints(self):
        cl = self.problem.c_lower
        cu = self.problem.c_upper
        # Adapt constraints a single/list of scipy LinearConstraint() or NonlinearConstraint() objects
        self.constraints = NonlinearConstraint(self.con, cl, cu, jac=self.jac)

    # # For callback, for every method except trust-constr
    # # trust-constr can call with more information
    # # Overrides base class update_outputs()
    # def update_outputs(self, xk, optimize_result=None):
    #     if len(self.options['readable_outputs']) > 0:
    #         name = self.problem_name
    #         with open(name + '_x.out', 'a') as f:
    #             np.savetxt(f, xk.reshape(1, xk.size))

    #     # For 'trust-constr', OptimizeResult() object state is available after each iteration
    #     if (optimize_result is not None) and (optimize_result != True):
    #         pass

    def print_results(self, optimal_variables=False):

        output  = "\n\tSolution from Scipy:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += f"\n\t{'Total function evals':25}: {self.results['nfev']}"
        if 'njev' in self.results:
            output += f"\n\t{'Gradient norm':25}: {np.linalg.norm(self.results['jac'])}"
            output += f"\n\t{'Total gradient evals':25}: {self.results['njev']}"
        if 'nit' in self.results:
            output += f"\n\t{'Major iterations':25}: {self.results['nit']}"
        if optimal_variables:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"

        output += '\n\t' + '-'*100
        print(output)