import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class LBFGSB(Optimizer):
    '''
    Class that interfaces modOpt with the L-BFGS-B optimization algorithm from Scipy.
    L-BFGS-B (Limited-memory BFGS with Bound constraints) 
    is a quasi-Newton optimization algorithm for large-scale bound-constrained problems.
    Therefore, it does not support other types of constraints.

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
    turn_off_outputs : bool, default=False
        If ``True``, prevents modOpt from generating any output files.

    solver_options : dict, default={}
        Dictionary containing the options to be passed to the solver.
        Available options are: 'maxfun', 'maxiter', 'maxls', 'maxcor',
        'ftol', 'gtol', 'iprint', 'callback'.
        See the LBFGSB page in modOpt's documentation for more information.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'x', 'obj'.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        self.solver_name = 'scipy-l-bfgs-b'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxfun': (int, 1000),  # Max num of function evaluations (default: 15000 in scipy.optimize.minimize)
            'maxiter': (int, 200),  # Max num of iterations (default:15000 in scipy.optimize.minimize)
            'maxls': (int, 20),     # Max num of line search steps (per major iteration)
            'maxcor': (int, 10),    # Maximum number of variable metric corrections used to define the limited memory Hessian approximation
            'ftol': (float, 2.22e-9),  # Terminate successfully if: `(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol`
            'gtol': (float, 1e-5),  # Terminate successfully if: `max{|proj g_i | i = 1, ..., n} <= gtol`, where `proj g_i` is the i-th component of the projected gradient.
            'iprint': (int, -1),    # Controls the frequency of output (<0, =0, 0<iprint<99, =99, =100, >100)
            'callback': ((type(None), callable), None),
        }

        # Used for verifying the keys and value-types of user-provided solver_options
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])
        
        # Declare outputs
        self.available_outputs = {
            'x'  : (float, (self.problem.nx,)),
            'obj': float
            }
        self.options.declare('readable_outputs', values=([], ['x'], ['obj'], ['x', 'obj']), default=[])

        # Define the initial guess, objective, gradient
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']

    def setup(self):
        '''
        Setup the optimizer.
        Setup outputs, bounds, and constraints.
        Check the validity of user-provided 'solver_options'.
        '''
        self.solver_options.update(self.options['solver_options'])
        self.options_to_pass = self.solver_options.get_pure_dict()
        self.user_callback = self.options_to_pass.pop('callback')
        self.setup_bounds()
        if self.problem.constrained:
            raise RuntimeError('LBFGSB does not support constraints. ' \
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove constraints.')

        # Check if gradient is declared and raise error/warning for Problem/ProblemLite
        self.check_if_callbacks_are_declared('grad', 'Objective gradient', 'LBFGSB')

    def setup_bounds(self):
        '''
        Adapt bounds as a Scipy Bounds() object.
        Only for  Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, trust-constr, COBYLA, and COBYQA methods.
        '''
        xl = self.problem.x_lower
        xu = self.problem.x_upper

        if np.all(xl == -np.inf) and np.all(xu == np.inf):
            self.bounds = None
        else:
            self.bounds = Bounds(xl, xu, keep_feasible=False)

    def solve(self):

        def callback(intermediate_result): 
            x = intermediate_result['x']
            f = intermediate_result['fun']
            self.update_outputs(x=x, obj=f)
            if self.user_callback: self.user_callback(x, f)

        self.update_outputs(x=self.x0, obj=self.obj(self.x0))

        # Call the L-BFGS-B algorithm from scipy (options are specific to L-BFGS-B)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method='L-BFGS-B',
            jac=self.grad,
            hess=None,
            hessp=None,
            bounds=self.bounds,
            constraints=None,
            tol=None,
            callback=callback,
            options=self.options_to_pass
            )
        self.total_time = time.time() - start_time

        self.results['hess_inv'] = self.results['hess_inv'].todense()

        self.run_post_processing()

        return self.results
    
    def print_results(self,
                      optimal_variables=False,
                      optimal_gradient=False,
                      optimal_hessian_inverse=False,
                      all=False):
        '''
        Print the optimization results to the console.

        Parameters
        ----------
        optimal_variables : bool, default=False
            If ``True``, print the optimal variables.
        optimal_gradient : bool, default=False
            If ``True``, print the optimal objective gradient.
        optimal_hessian_inverse : bool, default=False
            If ``True``, print the optimal Hessian inverse.
        all : bool, default=False
            If ``True``, print all available information.
        '''
        output  = "\n\tSolution from Scipy L-BFGS-B:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Status':25}: {self.results['status']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += f"\n\t{'Gradient norm':25}: {np.linalg.norm(self.results['jac'])}"
        output += f"\n\t{'Total function evals':25}: {self.results['nfev']}"
        output += f"\n\t{'Total gradient evals':25}: {self.results['njev']}"
        output += f"\n\t{'Major iterations':25}: {self.results['nit']}"
        output += self.get_callback_counts_string(25)
        
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if optimal_gradient or all:
            output += f"\n\t{'Optimal obj. gradient':25}: {self.results['jac']}"
        if optimal_hessian_inverse or all:
            output += f"\n\t{'Optimal Hessian inverse':25}: {self.results['hess_inv']}"

        output += '\n\t' + '-'*100
        print(output)