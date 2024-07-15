import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class BFGS(Optimizer):
    ''' 
    Class that interfaces modOpt with the BFGS optimization algorithm from Scipy.
    BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization algorithm 
    for unconstrained problems. Therefore, it does not support bounds or constraints.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        self.solver_name = 'scipy-bfgs'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 200),      # len(x0) * 200 is the default in scipy.optimize.minimize
            'gtol': (float, 1e-5),      # Gradient norm <= gtol for successful termination
            'xrtol': (float, 0.0),      # Relative tolerance for x. Terminate successfully if norm[alpha*pk] <= (norm[xk] + xrtol) * xrtol
            'norm': (float, np.inf),    # Order of gradient or step (alpha*pk) norm (inf is max, -inf is min)
            'c1': (float, 1e-4),        # Armijo condition parameter
            'c2': (float, 0.9),         # Curvature condition parameter, 0 < c1 < c2 < 1
            'hess_inv0': (np.ndarray, np.identity(self.problem.nx)), # Initial inverse Hessian approximation
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
        self.options.declare('readable_outputs', values=([], ['x'], ['obj'], ['x', 'obj']), default=[])

        # Define the initial guess, objective, gradient
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.acive_callbacks = ['obj', 'grad']

    def setup(self):
        '''
        Setup the optimizer.
        Setup outputs.
        Check the validity of user-provided 'solver_options'.
        '''
        self.solver_options.update(self.options['solver_options'])
        
        xl = self.problem.x_lower
        xu = self.problem.x_upper
        unbounded = (np.all(xl == -np.inf) and np.all(xu == np.inf))
        if not unbounded:
            raise RuntimeError('BFGS does not support bounds on variables. ' \
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove bounds.')
        
        if self.problem.constrained:
            raise RuntimeError('BFGS does not support constraints. ' \
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove constraints.')
        
        # Check if gradient is declared and raise error/warning for Problem/ProblemLite
        self.check_if_callbacks_are_declared('grad', 'Objective gradient', 'BFGS')

    def solve(self):
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback')

        def callback(intermediate_result):
            x = intermediate_result['x']
            f = intermediate_result['fun']
            self.update_outputs(x=x, obj=f)
            if user_callback: user_callback(x, f)

        self.update_outputs(x=self.x0, obj=self.obj(self.x0))

        # Call the BFGS algorithm from scipy (options are specific to BFGS)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method='BFGS',
            jac=self.grad,
            hess=None,
            hessp=None,
            bounds=None,
            constraints=None,
            tol=None,
            callback=callback,
            options=solver_options
            )
        self.total_time = time.time() - start_time

        self.run_post_processing()

        return self.results
    
    def print_results(self, 
                      optimal_variables=False,
                      optimal_gradient=False,
                      optimal_hessian_inverse=False,
                      all=False):
        '''
        Print the results of the optimization in modOpt's format.
        '''
        output  = "\n\tSolution from Scipy BFGS:"
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
        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if optimal_gradient or all:
            output += f"\n\t{'Optimal obj. gradient':25}: {self.results['jac']}"
        if optimal_hessian_inverse or all:
            output += f"\n\t{'Optimal Hessian inverse':25}: {self.results['hess_inv']}"

        output += '\n\t' + '-'*100
        print(output)