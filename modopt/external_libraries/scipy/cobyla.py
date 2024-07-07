import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class COBYLA(Optimizer):
    '''
    Class that interfaces modOpt with the COBYLA optimization algorithm from Scipy.
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
        self.options.declare('outputs', values=([],['x']), default=[])

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
        self.setup_outputs()
        self.solver_options.update(self.options['solver_options'])
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

        if xl.all() == -np.inf and xu.all() == np.inf:
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

        self.constraints = []
        if len(eqi) > 0:
            raise RuntimeError('Detected equality constraints in the problem. '\
                               'COBYLA does not support equality constraints. '\
                               'Use a different solver (PySLSQP, IPOPT, etc.) or remove the equality constraints.')

        if len(lci) > 0:
            con_dict_ineq1 = {}
            con_dict_ineq1['type'] = 'ineq'
            con_dict_ineq1['fun'] = lambda x: self.con(x)[lci] - cl[lci]
            self.constraints.append(con_dict_ineq1)

        if len(uci) > 0:
            con_dict_ineq2 = {}
            con_dict_ineq2['type'] = 'ineq'
            con_dict_ineq2['fun'] = lambda x: cu[uci] - self.con(x)[uci]
            self.constraints.append(con_dict_ineq2)

    def solve(self):
        method = 'COBYLA'
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback')

        def callback(x): 
            self.update_outputs(x=x)
            if user_callback: user_callback(x) 

        self.update_outputs(x=self.x0)

        # Call the COBYLA algorithm from scipy (options are specific to COBYLA)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method=method,
            jac=None,
            hess=None,
            hessp=None,
            bounds=self.bounds,
            constraints=self.constraints,
            tol=None,
            callback=callback,
            options=solver_options
            )
        self.total_time = time.time() - start_time

        return self.results
    
    def print_results(self, optimal_variables=False):
        '''
        Print the results of the optimization in modOpt's format.
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
        if optimal_variables:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"

        output += '\n\t' + '-'*100
        print(output)