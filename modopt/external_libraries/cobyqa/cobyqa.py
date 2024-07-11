import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint #, minimize
import warnings
try: 
    # import latest cobyqa if available
    from cobyqa import minimize
except ImportError:
    # else access cobyqa installed with Scipy>=1.14.0 (requires python>=3.10)
    try:
        from scipy._lib.cobyqa import minimize
    except ImportError:
        warnings.warn("'cobyqa' could not be imported. Install cobyqa using 'pip install cobyqa' for using COBYQA optimizer.")
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class COBYQA(Optimizer):
    ''' 
    Class that interfaces modOpt with the COBYQA optimization algorithm.
    Constrained Optimization BY Quadratic Approximations or COBYQA is a gradient-free optimization algorithm.
    Unlike COBYLA, COBYQA also supports equality constraints.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        # Declare options
        self.solver_name = 'cobyqa'
        self.options.declare('solver_options', types=dict, default={})
        NPT = int(2*self.problem.nx+1)
        self.default_solver_options = {
            'maxfev': (int, 500),               # Maximum number of function evaluations (default: 500 * len(x0))
            'maxiter': (int, 1000),             # Maximum number of iterations (default: 1000 * len(x0))
            'target': (float, -np.inf),         # Target objective function value
            'feasibility_tol': (float, 1e-8),   # Tolerance on constraint violations
            'radius_init': (float, 1.0),        # Initial trust region radius
            'radius_final': (float, 1e-6),      # Final trust region radius
            'nb_points': (int, NPT),            # Number of interpolation points used to build the quadratic model of f and c, 0<NPT<=(n+1)*(n+2)//2
            'scale': (bool, False),             # Whether to scale the variables according to the bounds
            'filter_size': (int, int(1e6)),     # Maximum number of points in the filter. The filter is used to store the best point returned by the algorithm
            'store_history': (bool, False),     # Whether to store the history of the function evaluations
            'history_size': (int, int(1e6)),    # Maximum number of function evaluations to store in the history
            'debug': (bool, False),             # To perform additional checks during the optimization procedure.
            'disp': (bool, False),
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
        self.options.declare('outputs', values=([], ['x'], ['obj'], ['x', 'obj']), default=[])

        # Define the initial guess, objective, constraints
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        if self.problem.constrained:
            self.con  = self.problem._compute_constraints

    def setup(self):
        '''
        Setup the optimizer.
        Setup outputs, bounds, and constraints.
        Check the validity of user-provided 'solver_options'.
        '''
        # Setup outputs to be written to file
        self.setup_outputs()
        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])
        # Adapt bounds as scipy Bounds() object
        self.setup_bounds()

        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        if self.problem.constrained:
            self.setup_constraints()
        else:
            self.constraints = ()

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
            # OR the following can also be used
            # self.bounds = np.array([xl, xu]).T

    def setup_constraints(self):
        '''
        Adapt constraints as a a single/list of scipy LinearConstraint() or NonlinearConstraint() objects.
        '''
        cl = self.problem.c_lower
        cu = self.problem.c_upper
        self.constraints = NonlinearConstraint(self.con, cl, cu)

    def solve(self):
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback')

        def callback(intermediate_result): 
            x = intermediate_result['x']
            f = intermediate_result['fun']
            self.update_outputs(x=x, obj=f)
            if user_callback: user_callback(x, f)

        # self.update_outputs(x=self.x0, obj=self.obj(self.x0)) # maybe required for scipy.optimize.minimize

        # Call the cobyqa algorithm (not from Scipy)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            # method='COBYQA',  # for scipy.optimize.minimize
            # jac=None,         # for scipy.optimize.minimize
            # hess=None,        # for scipy.optimize.minimize
            # hessp=None,       # for scipy.optimize.minimize
            bounds=self.bounds,
            constraints=self.constraints,
            # tol=None,         # for scipy.optimize.minimize
            callback=callback,
            options=solver_options
            )
        self.total_time = time.time() - start_time

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      obj_history=False,
                      max_con_viol_history=False):
        '''
        Print the results of the optimization in modOpt's format.
        '''
        output  = "\n\tSolution from COBYQA:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':25}: {self.problem_name}"
        output += f"\n\t{'Solver':25}: {self.solver_name}"
        output += f"\n\t{'Success':25}: {self.results['success']}"
        output += f"\n\t{'Message':25}: {self.results['message']}"
        output += f"\n\t{'Status':25}: {self.results['status']}"
        output += f"\n\t{'Total time':25}: {self.total_time}"
        output += f"\n\t{'Objective':25}: {self.results['fun']}"
        output += f"\n\t{'Max. constraint violation':25}: {self.results['maxcv']}"
        output += f"\n\t{'Total function evals':25}: {self.results['nfev']}"
        output += f"\n\t{'Total iterations':25}: {self.results['nit']}"
        if optimal_variables:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if obj_history and self.solver_options['store_history']:
            output += f"\n\t{'Objective history':25}: {self.results['fun_history']}"
        if max_con_viol_history and self.solver_options['store_history']:
            output += f"\n\t{'Max. con. viol. history':25}: {self.results['maxcv_history']}"

        output += '\n\t' + '-'*100
        print(output)
