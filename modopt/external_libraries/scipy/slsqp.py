import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary
from modopt import Optimizer

class SLSQP(Optimizer):
    ''' 
    Class that interfaces modOpt with the SLSQP optimization algorithm from Scipy.
    '''
    def initialize(self):
        '''
        Initialize the optimizer.
        Declare options, solver_options and outputs.
        '''
        # Declare options
        self.solver_name = 'scipy-slsqp'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 100),
            'ftol': (float, 1e-6),
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

        # Define the initial guess, objective, gradient, constraints, jacobian
        self.x0   = self.problem.x0 * 1.0
        self.obj  = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        if self.problem.constrained:
            self.con  = self.problem._compute_constraints
            self.jac  = self.problem._compute_constraint_jacobian

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

    def setup_constraints(self):
        '''
        Adapt constraints as a list of dictionaries with constraints =0 or  >= 0.
        '''
        cl = self.problem.c_lower
        cu = self.problem.c_upper

        eqi = np.where(cl == cu)[0]
        lci = np.where((cl != -np.inf) & (cl != cu))[0]
        uci = np.where((cu !=  np.inf) & (cl != cu))[0]

        self.constraints = []
        if len(eqi) > 0:
            con_dict_eq = {}
            con_dict_eq['type'] = 'eq'
            con_dict_eq['fun'] = lambda x: self.con(x)[eqi] - cl[eqi]
            con_dict_eq['jac'] = lambda x: self.jac(x)[eqi]
            self.constraints.append(con_dict_eq)

        if len(lci) > 0:
            con_dict_ineq1 = {}
            con_dict_ineq1['type'] = 'ineq'
            con_dict_ineq1['fun'] = lambda x: self.con(x)[lci] - cl[lci]
            con_dict_ineq1['jac'] = lambda x: self.jac(x)[lci]
            self.constraints.append(con_dict_ineq1)

        if len(uci) > 0:
            con_dict_ineq2 = {}
            con_dict_ineq2['type'] = 'ineq'
            con_dict_ineq2['fun'] = lambda x: cu[uci] - self.con(x)[uci]
            con_dict_ineq2['jac'] = lambda x: -self.jac(x)[uci]
            self.constraints.append(con_dict_ineq2)

    def solve(self):
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback')

        def callback(x): 
            self.update_outputs(x=x)
            if user_callback: user_callback(x) 

        self.update_outputs(x=self.x0)

        # Call the SLSQP algorithm from scipy (options are specific to SLSQP)
        start_time = time.time()
        self.results = minimize(
            self.obj,
            self.x0,
            args=(),
            method='SLSQP',
            jac=self.grad,
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
    
    def print_results(self, 
                      optimal_variables=False,
                      optimal_gradient=False):
        '''
        Print the results of the optimization in modOpt's format.
        '''
        output  = "\n\tSolution from Scipy SLSQP:"
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
        if optimal_variables:
            output += f"\n\t{'Optimal variables':25}: {self.results['x']}"
        if optimal_gradient:
            output += f"\n\t{'Optimal obj. gradient':25}: {self.results['jac']}"

        output += '\n\t' + '-'*100
        print(output)