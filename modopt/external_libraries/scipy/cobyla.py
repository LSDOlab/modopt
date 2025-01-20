import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class COBYLA(Optimizer):
    '''
    Class that interfaces modOpt with the COBYLA optimization algorithm from Scipy.
    Constrained Optimization BY Linear Approximations or COBYLA is a gradient-free optimization algorithm.
    COBYLA only supports inequality constraints and bounds.

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
        Available options are: 'maxiter', 'rhobeg', 'tol', 'catol', 'disp', 'callback'.
        See the COBYLA page in modOpt's documentation for more information.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'x'.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        self.solver_name = 'scipy-cobyla'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 1000), # Maximum number of function evaluations
            'rhobeg': (float, 1.0), # Reasonable initial changes to the variables
            'tol': (float, 1e-4),   # Final accuracy in the optimization (lower bound on the size of the trust region)
            'catol': (float, 2e-4), # Absolute constraint violation tolerance
            'disp': (bool, False),
            'callback': ((type(None), callable), None),
        }

        # Used for verifying the keys and value-types of user-provided solver_options
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])
        
        # Declare outputs
        self.available_outputs = {'x': (float, (self.problem.nx,))}
        self.options.declare('readable_outputs', values=([],['x']), default=[])

        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.active_callbacks = ['obj']
        if self.problem.constrained:
            self.con  = self.problem._compute_constraints
            self.active_callbacks += ['con']

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
            self.setup_constraints()
        else:
            self.constraints = ()

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

    def setup_constraints(self):
        '''
        Adapt constraints as a list of dictionaries with constraints >= 0.
        Note: COBYLA only supports inequality constraints.
        Raises
        ------
        RuntimeError
            If equality constraints are detected in the problem.
        '''
        cl = self.problem.c_lower
        cu = self.problem.c_upper

        eqi = np.where(cl == cu)[0]
        lci = np.where((cl != -np.inf) & (cl != cu))[0]
        uci = np.where((cu !=  np.inf) & (cl != cu))[0]

        if len(eqi) > 0:
            raise RuntimeError('Detected equality constraints in the problem. '\
                               'COBYLA does not support equality constraints. '\
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove the equality constraints.')
        
        # problem is constrained (with no equalities), set up constraints list of dictionaries
        def fun(x):
            c = self.con(x)
            return np.concatenate((c[lci] - cl[lci], cu[uci] - c[uci]))
        self.constraints = ({'type': 'ineq', 'fun': fun}, )

    def solve(self):

        def callback(x): 
            self.update_outputs(x=x)
            if self.user_callback: self.user_callback(x) 

        self.update_outputs(x=self.x0)

        # Call the COBYLA algorithm from scipy (options are specific to COBYLA)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method='COBYLA',
            jac=None,
            hess=None,
            hessp=None,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=None,
            callback=callback,
            options=self.options_to_pass
            )
        self.total_time = time.time() - start_time

        self.run_post_processing()

        return self.results
    
    def print_results(self, optimal_variables=False, all=False):
        '''
        Print the optimization results to the console.

        Parameters
        ----------
        optimal_variables : bool, default=False
            If ``True``, print the optimal variables.
        all : bool, default=False
            If ``True``, print all available outputs.
        '''
        output  = "\n\tSolution from Scipy COBYLA:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Status':25}: {self.results['status']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += f"\n\t{'Total function evals':25}: {self.results['nfev']}"
        output += f"\n\t{'Max. constraint violation':25}: {self.results['maxcv']}"
        output += self.get_callback_counts_string(25)
        
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"

        output += '\n\t' + '-'*100
        print(output)