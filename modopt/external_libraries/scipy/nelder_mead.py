import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer
from typing import Callable

class NelderMead(Optimizer):
    '''
    Class that interfaces modOpt with the Nelder-Mead optimization algorithm from Scipy.
    Nelder-Mead is a gradient-free optimization algorithm that can solve bound-constrained problems.
    It does not support other types of constraints.
    
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
        Available options are: 'maxiter', 'maxfev', 'xatol', 'fatol', 'adaptive',
        'initial_simplex', 'return_all', 'disp', 'callback'.
        See the NelderMead page in modOpt's documentation for more information.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'x', 'obj'.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        self.solver_name = 'scipy-nelder-mead'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 1000),     # Max num of iterations (default:len(x)*200 in scipy.optimize.minimize)
            'maxfev': (int, 1000),      # Max num of function evaluations (default:len(x)*200 in scipy.optimize.minimize)
            'xatol': (float, 1e-4),     # Absolute error in xopt between iterations that is acceptable for convergence.
            'fatol': (float, 1e-4),     # Absolute error in func(xopt) between iterations that is acceptable for convergence.
            'adaptive': (bool, False),  # Adapt algorithm parameters to dimensionality of problem. Useful for high-dimensional minimization.
            'initial_simplex': ((type(None), np.ndarray), None), # Initial simplex
            'return_all': (bool, False),# To return a list of the best solution at each major iteration in the final results dict
            'disp': (bool, False),
            'callback': ((type(None), Callable), None),
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

        # Define the initial guess, objective
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.active_callbacks = ['obj']

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
            raise RuntimeError('NelderMead does not support constraints. ' \
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove constraints.')

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

        # Call the Nelder-Mead algorithm from scipy (options are specific to Nelder-Mead)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method='Nelder-Mead',
            jac=None,
            hess=None,
            hessp=None,
            bounds=self.bounds,
            constraints=None,
            tol=None,
            callback=callback,
            options=self.options_to_pass
            )
        self.total_time = time.time() - start_time

        self.results['final_simplex'] = {
            'coordinates': self.results['final_simplex'][0],
            'obj_values': self.results['final_simplex'][1]
            }
        
        self.run_post_processing()

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      final_simplex=False,
                      all=False):
        '''
        Print the optimization results to the console.

        Parameters
        ----------
        optimal_variables : bool, default=False
            If ``True``, print the optimal variables.
        final_simplex : bool, default=False
            If ``True``, print the final simplex coordinates and their objective values.
        all : bool, default=False
            If ``True``, print all available information.
        '''
        output  = "\n\tSolution from Scipy Nelder-Mead:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Status':25}: {self.results['status']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += f"\n\t{'Total function evals':25}: {self.results['nfev']}"
        output += f"\n\t{'Total iterations':25}: {self.results['nit']}"
        output += self.get_callback_counts_string(25)

        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if final_simplex or all:
            output += f"\n\t{'Final simplex':25}: {self.results['final_simplex']}"

        output += '\n\t' + '-'*100
        print(output)