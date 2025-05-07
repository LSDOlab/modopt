import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, BFGS
from scipy.sparse import coo_array
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer
from typing import Callable

class TrustConstr(Optimizer):
    ''' 
    Class that interfaces modOpt with the trust-constr optimization algorithm from Scipy.
    The trust-constr algorithm uses a trust-region interior point method or 
    an equality-constrained sequential quadratic programming (SQP) method
    to solve a problem depending on whether the problem has inequality constraints or not.
    It can make use of second order information in the form of the Hessian of 
    the objective for unconstrained problems or the Hessian of the Lagrangian for constrained 
    problems. TrustConstr can also use objective HVP (Hessian-vector product) when the
    objective Hessian is unavailable.

    Parameters
    ----------
    problem : Problem or ProblemLite
        Object containing the problem to be solved.
    recording : bool, default=False
        If ``True``, record all outputs from the optimization.
        This needs to be enabled for hot-starting the same problem later,
        if the optimization is interrupted.
    hot_start_from : str, optional
        The record file from which to hot-start the optimization.
    hot_start_atol : float, default=0.
        The absolute tolerance check for the inputs
        when reusing outputs from the hot-start record.
    hot_start_rtol : float, default=0.
        The relative tolerance check for the inputs
        when reusing outputs from the hot-start record.
    visualize : list, default=[]
        The list of scalar variables to visualize during the optimization.
    keep_viz_open : bool, default=False
        If ``True``, keep the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    solver_options : dict, default={}
        Dictionary containing the options to be passed to the solver.
        Available options are: 'maxiter', 'gtol', 'xtol', 'barrier_tol', 'initial_tr_radius',
        'initial_constr_penalty', 'initial_barrier_parameter', 'initial_barrier_tolerance',
        'factorization_method', 'sparse_jacobian', 'ignore_exact_hessian', 'verbose', 'callback'.
        See the TrustConstr page in modOpt's documentation for more information.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'x', 'obj', 'opt', 'feas', 'grad', 'lgrad', 'con', 'jac',
        'lmult_x', 'lmult_c', 'iter', 'cg_niter', 'nfev', 'nfgev', 'nfhev', 'ncev', 'ncgev',
        'nchev', 'tr_radius', 'constr_penalty', 'barrier_parameter', 'barrier_tolerance',
        'cg_stop_cond', 'time'.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        # Declare options
        self.solver_name = 'scipy-trust-constr'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 500),            # Maximum number of iterations
            'gtol': (float, 1e-8),            # Terminate successfully when both the inf norm (max abs value) of the Lag. gradient 
                                              # and the con. violation are less than `gtol`.
            'xtol': (float, 1e-8),            # Terminate successfully when `tr_radius < xtol`
            'barrier_tol': (float, 1e-8),     # Terminate successfully when the barrier parameter decays below `barrier_tol`
            'initial_tr_radius': (float, 1.),           # Initial trust region radius
            'initial_constr_penalty': (float, 1.),      # Initial constraints penalty parameter for the merit function
            'initial_barrier_parameter': (float, 0.1),  # Initial barrier parameter
            'initial_barrier_tolerance': (float, 0.1),  # Initial tolerance for the barrier subproblem termination
            'factorization_method': ((type(None), str), None, ('NormalEquation', 'AugmentedSystem', 'QRFactorization', 'SVDFactorization', None)),
                                                            # Method to use for factorizing the Jacobian matrix.
            'sparse_jacobian': ((type(None), bool), None),  # All constraints must have the same kind of the Jacobian - either all sparse or all dense. "
                                                            # You can set the sparsity globally by setting `sparse_jacobian` True of False."
            'ignore_exact_hessian': (bool, False),  # To ignore exact hessian and use only gradient information to approximate the hessian
            'verbose': (int, 0, (0,1,2,3)),         # Verbosity level
            'callback': ((type(None), Callable), None),
        }

        # Used for verifying the keys and value-types of user-provided solver_options
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])

        # Declare outputs
        self.available_outputs = {
            'x'  : (float, (self.problem.nx,)),
            'obj': float,
            'opt': float,
            'feas': float,
            'grad': (float, (self.problem.nx,)),
            'lgrad': (float, (self.problem.nx,)),
            'con': (float, (self.problem.nc,)),
            'jac': (float, (self.problem.nc, self.problem.nx)),
            'lmult_x': (float, (self.problem.nx,)),
            'lmult_c': (float, (self.problem.nc,)),
            'iter': int,
            'cg_niter': int,
            'nfev': int,
            'nfgev': int,
            'nfhev': int,
            'ncev': int,
            'ncgev': int,
            'nchev': int,
            'tr_radius': float,
            'constr_penalty': float,
            'barrier_parameter': float,
            'barrier_tolerance': float,
            'cg_stop_cond': float,
            'time': float,
            }
        self.options.declare('readable_outputs', types=list, default=[])

        # Define the initial guess, objective, gradient, constraints, jacobian
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con  = self.problem._compute_constraints
            self.jac  = self.problem._compute_constraint_jacobian
            self.active_callbacks += ['con', 'jac']

    def setup(self):
        '''
        Setup the optimizer.
        Setup outputs, bounds, and constraints.
        Check the validity of user-provided 'solver_options'.
        '''
        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])
        self.options_to_pass = self.solver_options.get_pure_dict()
        self.options_to_pass.pop('ignore_exact_hessian')
        self.user_callback = self.options_to_pass.pop('callback')
        # Adapt bounds as scipy Bounds() object
        self.setup_bounds()

        # Set up Hessians
        self.obj_hess = None
        self.obj_hvp = None
        # self.con_hess = None
        # NOTE: Fix to use BFGS as the constraint Hessian since Scipy does not take `None` as said in their docs
        #       Scipy default for constraint Hessian is BFGS() but for obj Hessian 'hess' and 'hessp' is None
        # NOTE: This was fixed in Scipy with a commit on 18 July 2024
        self.con_hess = BFGS()

        if not self.solver_options['ignore_exact_hessian']:
            if not self.problem.constrained:
                if 'obj_hess' in self.problem.user_defined_callbacks:
                    self.obj_hess = self.problem._compute_objective_hessian
                    self.active_callbacks += ['obj_hess']
                elif 'obj_hvp' in self.problem.user_defined_callbacks: # use hvp only if hess is not available
                    self.obj_hvp = self.problem._compute_objective_hvp
                    self.active_callbacks += ['obj_hvp']
            else:
                if 'lag_hess' in self.problem.user_defined_callbacks:
                    # NOTE: Hack to use Lagrangian Hessian instead of the weighted sum of just the constraint Hessians without the objective Hessian
                    #       This works because the trust-constr algorithm computes Lagrangian Hessian as the sum of the objective Hessian and the weighted sum of constraint Hessians.
                    #       So we pass the obj_hess as zeros((n,n)) and the weighted sum of the constraint Hessians as the Lagrangian Hessian.
                    #       This could BREAK in the future if the scipy trust-constr algorithm changes.
                    self.obj_hess = lambda x: coo_array((self.problem.nx, self.problem.nx), dtype=np.float64)
                    self.con_hess = lambda x, v: self.problem._compute_lagrangian_hessian(x, v)
                    self.active_callbacks += ['lag_hess']
        
        # Set up constraints
        if self.problem.constrained:
            self.setup_constraints()
            self.tr_interior_point = bool(self.bounds) or self.ineq_constrained
        else:
            self.constraints = ()
            self.tr_interior_point = bool(self.bounds)

    def setup_bounds(self):
        '''
        Adapt bounds as a Scipy Bounds() object.
        Only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, COBYLA, and COBYQA methods.
        '''
        xl = self.problem.x_lower
        xu = self.problem.x_upper
        if np.all(xl == -np.inf) and np.all(xu == np.inf):
            self.bounds = None
        else:
            self.bounds = Bounds(xl, xu, keep_feasible=False)

    def setup_constraints(self):
        '''
        Adapt constraints as a a single/list of scipy LinearConstraint() or NonlinearConstraint() objects.
        '''
        cl = self.problem.c_lower
        cu = self.problem.c_upper
        self.constraints = NonlinearConstraint(self.con, cl, cu, jac=self.jac, hess=self.con_hess, keep_feasible=False)
        # self.constraints = NonlinearConstraint(self.con, cl, cu, jac=self.jac, keep_feasible=self.solver_options.pop('keep_feasible'))

        lci = np.where((cl != -np.inf) & (cl != cu))[0]
        uci = np.where((cu !=  np.inf) & (cl != cu))[0]
        self.ineq_constrained = True if len(lci) + len(uci) > 0 else False

    def solve(self):
        constrained = self.problem.constrained
        bounded     = bool(self.bounds)
        tr_ip = self.tr_interior_point
        def callback(intermediate_result):   # 23(=26-3) in total niter/nit are the same, method/status same until termination
            # print("Intermediate result: ") # total 28 keys in final_results including extra 'message'/'success' keys and duplicate 'nit'/'niter' keys
            # print(intermediate_result)
            con     = intermediate_result['constr'][0] if constrained else []
            jac     = intermediate_result['jac'][0]    if constrained else []
            lmult_c = intermediate_result['v'][0]      if constrained else []
            lmult_x = intermediate_result['v'][-1]     if bounded     else []

            ncev  = intermediate_result['constr_nfev'][0] if constrained else 0
            ncgev = intermediate_result['constr_njev'][0] if constrained else 0
            nchev = intermediate_result['constr_nhev'][0] if constrained else 0

            barrier_parameter = intermediate_result['barrier_parameter'] if tr_ip else 0.0
            barrier_tolerance = intermediate_result['barrier_tolerance'] if tr_ip else 0.0
            self.update_outputs(
                x=intermediate_result['x'],
                opt=intermediate_result['optimality'],
                feas=intermediate_result['constr_violation'],
                obj=intermediate_result['fun'],
                grad=intermediate_result['grad'],
                lgrad=intermediate_result['lagrangian_grad'],
                con=con,
                jac=jac,
                lmult_c=lmult_c,
                lmult_x=lmult_x,
                iter=intermediate_result['nit'],
                cg_niter=intermediate_result['cg_niter'],
                nfev=intermediate_result['nfev'],
                nfgev=intermediate_result['njev'],
                nfhev=intermediate_result['nhev'],
                ncev=ncev,
                ncgev=ncgev,
                nchev=nchev,
                tr_radius=intermediate_result['tr_radius'],
                constr_penalty=intermediate_result['constr_penalty'],
                barrier_parameter=barrier_parameter,
                barrier_tolerance=barrier_tolerance,
                cg_stop_cond=intermediate_result['cg_stop_cond'],
                time=intermediate_result['execution_time']
                )
            if self.user_callback: self.user_callback(intermediate_result)

        # Call the trust-constr algorithm from scipy (options are specific to trust-constr)
        start_time = time.time()
        final_result = minimize(
            self.obj,
            self.x0,
            args=(),
            method='trust-constr',
            jac=self.grad,
            hess=self.obj_hess,
            hessp=self.obj_hvp,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=None,
            callback=callback,
            options=self.options_to_pass
            )
        self.total_time = time.time() - start_time

        # Replace the scipy results with modOpt's results dictionary
        self.results = self.out_dict
        self.results['success'] = final_result.success
        self.results['message'] = final_result.message
        self.results['status']  = final_result.status
        self.results['method']  = final_result.method

        self.run_post_processing()

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      optimal_gradient=False,
                      optimal_constraints=False,
                      optimal_constraint_jacobian=False,
                      optimal_lagrange_multipliers=False,
                      optimal_lagrangian_gradient=False,
                      all=False):
        '''
        Print the optimization results to the console.

        Parameters
        ----------
        optimal_variables : bool, default=False
            If ``True``, print the optimal variables.
        optimal_gradient : bool, default=False
            If ``True``, print the optimal objective gradient.
        optimal_constraints : bool, default=False
            If ``True``, print the optimal constraints.
        optimal_constraint_jacobian : bool, default=False
            If ``True``, print the optimal constraints Jacobian.
        optimal_lagrange_multipliers : bool, default=False
            If ``True``, print the optimal Lagrange multipliers.
        optimal_lagrangian_gradient : bool, default=False
            If ``True``, print the optimal Lagrangian gradient.
        all : bool, default=False
            If ``True``, print all available information.
        '''
        constrained = self.problem.constrained
        bounded     = bool(self.bounds)
        tr_ip       = self.tr_interior_point

        # con     = self.results['constr'][0] if constrained else []
        # jac     = self.results['jac'][0]    if constrained else []
        # lmult_c = self.results['v'][0]      if constrained else []
        # lmult_x = self.results['v'][-1]     if bounded     else []

        # ncev  = self.results['constr_nfev'][0] if constrained else []
        # ncgev = self.results['constr_njev'][0] if constrained else []
        # nchev = self.results['constr_nhev'][0] if constrained else []

        barrier_parameter = self.results['barrier_parameter'] if tr_ip else 'None (since using equality_constrained_sqp)'
        barrier_tolerance = self.results['barrier_tolerance'] if tr_ip else 'None (since using equality_constrained_sqp)'

        output  = "\n\tSolution from Scipy trust-constr:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':30}: {self.problem_name}"
        output += f"\n\t{'Solver':30}: {self.solver_name}"
        output += f"\n\t{'Method':30}: {self.results['method']}" # either ‘equality_constrained_sqp’ or ‘tr_interior_point’
        output += f"\n\t{'Success':30}: {self.results['success']}"
        output += f"\n\t{'Message':30}: {self.results['message']}"
        output += f"\n\t{'Status':30}: {self.results['status']}"
        output += f"\n\t{'Total time':30}: {self.total_time}"
        output += f"\n\t{'Objective':30}: {self.results['obj']}"
        output += f"\n\t{'Gradient norm':30}: {np.linalg.norm(self.results['grad'])}" # extra info not in keys of results
        output += f"\n\t{'Optimality':30}: {np.linalg.norm(self.results['opt'])}"
        output += f"\n\t{'Max. constr. violation':30}: {np.linalg.norm(self.results['feas'])}"
        output += f"\n\t{'Trust region radius':30}: {self.results['tr_radius']}"
        output += f"\n\t{'Constraint penalty':30}: {self.results['constr_penalty']}"
        output += f"\n\t{'Barrier parameter':30}: {barrier_parameter}"
        output += f"\n\t{'Barrier tolerance':30}: {barrier_tolerance}"
        output += f"\n\t{'Total function evals':30}: {self.results['nfev']}"
        output += f"\n\t{'Total gradient evals':30}: {self.results['nfgev']}"
        output += f"\n\t{'Total Hessian evals':30}: {self.results['nfhev']}"
        output += f"\n\t{'Total constraint evals':30}: {self.results['ncev']}"
        output += f"\n\t{'Total constr. Jacobian evals':30}: {self.results['ncgev']}"
        output += f"\n\t{'Total constr. Hessian evals':30}: {self.results['nchev']}"
        output += f"\n\t{'Total iterations':30}: {self.results['iter']}"
        output += f"\n\t{'CG iterations':30}: {self.results['cg_niter']}"
        output += f"\n\t{'CG stop condition':30}: {self.results['cg_stop_cond']}"
        output += self.get_callback_counts_string(30)

        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':30}: {self.results['x']}"
        if optimal_gradient or all:
            output += f"\n\t{'Optimal obj. gradient':30}: {self.results['grad']}"
        if optimal_constraints or all:
            output += f"\n\t{'Optimal constraints':30}: {self.results['con']}"
        if optimal_constraint_jacobian or all:
            output += f"\n\t{'Optimal con. Jacobian':30}: {self.results['jac']}"
        if optimal_lagrange_multipliers or all:
            output += f"\n\t{'Optimal Lag. mult. (bounds)':30}: {self.results['lmult_x']}"
            output += f"\n\t{'Optimal Lag. mult. (constr.)':30}: {self.results['lmult_c']}"
        if optimal_lagrangian_gradient or all:
            output += f"\n\t{'Optimal Lag. gradient':30}: {self.results['lgrad']}"

        output += '\n\t' + '-'*100
        print(output)