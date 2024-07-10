import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class NelderMead(Optimizer):
    '''
    Class that interfaces modOpt with the Nelder-Mead optimization algorithm from Scipy.
    Nelder-Mead is a gradient-free optimization algorithm that can solve bound-constrained problems.
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

        # Define the initial guess, objective
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective

    def setup(self):
        '''
        Setup the optimizer.
        Setup outputs, bounds, and constraints.
        Check the validity of user-provided 'solver_options'.
        '''
        self.setup_outputs()
        self.solver_options.update(self.options['solver_options'])
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
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback')

        def callback(intermediate_result): 
            x = intermediate_result['x']
            f = intermediate_result['fun']
            self.update_outputs(x=x, obj=f)
            if user_callback: user_callback(x, f)

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
            options=solver_options
            )
        self.total_time = time.time() - start_time

        self.results['final_simplex'] = {
            'coordinates': self.results['final_simplex'][0],
            'obj_values': self.results['final_simplex'][1]
            }

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      final_simplex=False):
        '''
        Print the results of the optimization in modOpt's format.
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
        if optimal_variables:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if final_simplex:
            output += f"\n\t{'Final simplex':25}: {self.results['final_simplex']}"

        output += '\n\t' + '-'*100
        print(output)