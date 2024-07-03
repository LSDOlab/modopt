import numpy as np
from modopt import Optimizer
import warnings
import time
from modopt.utils.options_dictionary import OptionsDictionary

try:
    from pyslsqp import optimize
except:
    warnings.warn("'pyslsqp' could not be imported. Install pyslsqp using 'pip install pyslsqp'.")

class PySLSQP(Optimizer):
    '''
    Class that interfaces modOpt with the PySLSQP package which is
    a Python wrapper for the SLSQP optimization algorithm.
    PySLSQP can solve nonlinear programming problems with
    equality and inequality constraints.
    '''
    def initialize(self, ):
        self.solver_name = 'pyslsqp'
        
        # No outputs can be declared for PySLSQP
        self.available_outputs = {}
        self.options.declare('outputs', values=([],), default=[])
        self.options.declare('solver_options', types=dict, default={})

        self.default_solver_options = {
            'maxiter': (int, 100),
            'acc': (float, 1e-6),
            'iprint': (int, 1),
            'callback': ((type(None), callable), None),
            'summary_filename': (str, 'slsqp_summary.out'),
            'visualize': (bool, False),
            'visualize_vars': (list, ['objective', 'optimality', 'feasibility']),
            'keep_plot_open': (bool, False),
            'save_figname': (str, 'slsqp_plot.pdf'),
            'save_itr': ((type(None), str), None),
            'save_vars': (list, ['x', 'objective', 'optimality', 'feasibility', 'step', 'iter', 'majiter', 'ismajor', 'mode']),
            'save_filename': (str, 'slsqp_recorder.hdf5'),
            'load_filename': ((type(None), str), None),
            'warm_start': (bool, False),
            'hot_start': (bool, False),
        }
        # Used for verifying the keys and value-types of user-provided solver_options, 
        # and generating an updated pure Python dictionary to provide pyslsqp.optimize()
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.con_in = self.problem._compute_constraints
        self.jac_in = self.problem._compute_constraint_jacobian

    def setup(self):
        '''
        Setup the initial guess, and matrices and vectors.
        '''
        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])

        self.x0 = self.problem.x0 * 1.
        self.nx = self.problem.nx * 1
        if self.problem.constrained:
            self.setup_constraints()

    def setup_constraints(self, ):
        cl = self.problem.c_lower
        cu = self.problem.c_upper

        eqi = self.eq_constraint_indices = np.where(cl == cu)[0]
        lci = self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
        uci = self.upper_constraint_indices = np.where((cu !=  np.inf) & (cl != cu))[0]

        self.eq_constrained    = True if len(eqi) > 0 else False
        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc_e = len(eqi)
        self.nc_i = len(lci) + len(uci)
        self.nc   = self.nc_e + self.nc_i

        if self.nc == 0:
            raise ValueError('No constraints found, but problem.constrained is set as True.')

    def con(self, x):
        c_in = self.con_in(x)

        eqi = self.eq_constraint_indices
        lci = self.lower_constraint_indices
        uci = self.upper_constraint_indices
        nc_e = self.nc_e

        c = np.zeros(self.nc)
        if self.eq_constrained:
            c[:nc_e] = c_in[eqi] - self.problem.c_lower[eqi]
        if self.lower_constrained:
            c[nc_e:nc_e + len(lci)] = c_in[lci] - self.problem.c_lower[lci]
        if self.upper_constrained:
            c[nc_e + len(lci):] = self.problem.c_upper[uci] - c_in[uci]

        return c
    
    def jac(self, x):
        j_in = self.jac_in(x)

        eqi = self.eq_constraint_indices
        lci = self.lower_constraint_indices
        uci = self.upper_constraint_indices
        nc_e = self.nc_e

        j = np.zeros((self.nc, self.nx))
        if self.eq_constrained:
            j[:nc_e] = j_in[eqi]
        if self.lower_constrained:
            j[nc_e:nc_e + len(lci)] = j_in[lci]
        if self.upper_constrained:
            j[nc_e + len(lci):] = -j_in[uci]

        return j


    def solve(self):
        # Set up the problem
        x0 = self.x0
        obj = self.obj
        grad = self.grad
        if self.problem.constrained:
            con = self.con
            jac = self.jac
        else:
            con = None
            jac = None

        xl = self.problem.x_lower
        xu = self.problem.x_upper
        meq = self.nc_e if self.problem.constrained else 0
        solver_options = self.solver_options.get_pure_dict() # = self.options['solver_options']

        # Run the optimization
        start_time = time.time()
        self.results = optimize(x0, obj=obj, con=con, grad=grad, jac=jac, xl=xl, xu=xu, meq=meq, **solver_options)
        self.total_time = time.time() - start_time
        
        return self.results
    
    def print_results(self, **kwargs):
        warnings.warn('PySLSQP prints the final results by default. '
                      'To suppress the results, set `solver_options={"iprint":0}`. '
                      'Check the summary file "slsqp_summary.out" for the summary of optimization.')
