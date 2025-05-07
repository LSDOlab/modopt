import numpy as np
from modopt import Optimizer
import time
from modopt.utils.options_dictionary import OptionsDictionary
from typing import Callable

class PySLSQP(Optimizer):
    '''
    Class that interfaces modOpt with the PySLSQP package which is
    a Python wrapper for the SLSQP optimization algorithm.
    PySLSQP can solve nonlinear programming problems with
    equality and inequality constraints.

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
        Available options are: 'maxiter', 'acc', 'iprint', 'callback', 'summary_filename',
        'visualize', 'visualize_vars', 'keep_plot_open', 'save_figname', 'save_itr',
        'save_vars', 'save_filename', 'load_filename', 'warm_start', 'hot_start'.
        See the PySLSQP page in modOpt's documentation for more information.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'x'.
    '''
    def initialize(self, ):
        self.solver_name = 'pyslsqp'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 100),
            'acc': (float, 1e-6),
            'iprint': (int, 1),
            'callback': ((type(None), Callable), None),
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

        # Declare outputs
        self.available_outputs = {'x': (float, (self.problem.nx,)),}
        self.options.declare('readable_outputs', values=([],['x']), default=[])

        self.x0 = self.problem.x0 * 1.
        self.nx = self.problem.nx * 1
        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']
        if self.problem.constrained:
            self.con_in = self.problem._compute_constraints
            self.jac_in = self.problem._compute_constraint_jacobian
            self.active_callbacks += ['con', 'jac']

    def setup(self):
        '''
        Setup the initial guess, and matrices and vectors.
        '''
        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])
        self.options_to_pass = self.solver_options.get_pure_dict()
        self.user_callback = self.options_to_pass.pop('callback')
        if hasattr(self, 'out_dir'):
            self.options_to_pass['summary_filename'] = self.out_dir + '/' + self.options_to_pass['summary_filename']
            self.options_to_pass['save_filename']    = self.out_dir + '/' + self.options_to_pass['save_filename']

        if self.problem.constrained:
            self.setup_constraints()

    def setup_constraints(self, ):
        cl = self.problem.c_lower
        cu = self.problem.c_upper

        eqi = self.eq_constraint_indices    = np.where(cl == cu)[0]
        lci = self.lower_constraint_indices = np.where((cl != -np.inf) & (cl != cu))[0]
        uci = self.upper_constraint_indices = np.where((cu !=  np.inf) & (cl != cu))[0]

        self.eq_constrained    = True if len(eqi) > 0 else False
        self.lower_constrained = True if len(lci) > 0 else False
        self.upper_constrained = True if len(uci) > 0 else False

        self.nc_e = len(eqi)
        self.nc_i = len(lci) + len(uci)
        self.nc   = self.nc_e + self.nc_i

        if self.nc == 0:
            raise ValueError('No constraint bounds found, but problem.constrained is set as True.')

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
        try:
            from pyslsqp import optimize
        except ImportError:
            raise ImportError("PySLSQP could not be imported or is not installed. Install it with 'pip install pyslsqp'.")

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
        
        def callback(x):
            self.update_outputs(x=x)
            if self.user_callback: self.user_callback(x) 

        self.update_outputs(x=self.x0)

        # Run the optimization
        start_time = time.time()
        self.results = optimize(x0, obj=obj, con=con, grad=grad, jac=jac, 
                                xl=xl, xu=xu, meq=meq, callback=callback, 
                                **self.options_to_pass)
        self.total_time = time.time() - start_time

        self.run_post_processing()

        return self.results
    
    def print_results(self,
                      optimal_variables=False,
                      optimal_gradient=False,
                      optimal_constraints=False,
                      optimal_jacobian=False,
                      optimal_multipliers=False,
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
        optimal_jacobian : bool, default=False
            If ``True``, print the optimal constraint Jacobian.
        optimal_multipliers : bool, default=False
            If ``True``, print the optimal multipliers.
        all : bool, default=False
            If ``True``, print all available information.
        '''
        
        output  = "\n\tSolution from PySLSQP:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':30}: {self.problem_name}"
        output += f"\n\t{'Solver':30}: {self.solver_name}"
        output += f"\n\t{'Success':30}: {self.results['success']}"
        output += f"\n\t{'Message':30}: {self.results['message']}"
        output += f"\n\t{'Status':30}: {self.results['status']}"
        output += f"\n\t{'Objective':30}: {self.results['objective']}"
        output += f"\n\t{'Optimality':30}: {self.results['optimality']}"
        output += f"\n\t{'Feasibility':30}: {self.results['feasibility']}"
        output += f"\n\t{'Gradient norm':30}: {np.linalg.norm(self.results['gradient'])}"
        output += f"\n\t{'Major iterations':30}: {self.results['num_majiter']}"
        output += f"\n\t{'Total function evals':30}: {self.results['nfev']}"
        output += f"\n\t{'Total gradient evals':30}: {self.results['ngev']}"
        output += f"\n\t{'Total time':30}: {self.total_time}"
        output += f"\n\t{'Function eval. time':30}: {self.results['fev_time']}"
        output += f"\n\t{'Derivative eval. time':30}: {self.results['gev_time']}"
        output += f"\n\t{'Optimizer time':30}: {self.results['optimizer_time']}"
        output += f"\n\t{'Processing time':30}: {self.results['processing_time']}"
        output += f"\n\t{'Visualization time':30}: {self.results['visualization_time']}"
        output += f"\n\t{'Summary saved in':30}: {self.results['summary_filename']}"
        if 'save_filename' in self.results.keys():
            output += f"\n\t{'Iteration data saved in':30}: {self.results['save_filename']}"
        if 'plot_filename' in self.results.keys():
            output += f"\n\t{'Plot saved in':30}: {self.results['plot_filename']}"
        if 'nfev_reused_in_hotstart' in self.results.keys():
            output += f"\n\t{'Fun. evals reused (hotstart)':30}: {self.results['nfev_reused_in_hotstart']}"
        if 'ngev_reused_in_hotstart' in self.results.keys():
            output += f"\n\t{'Der. evals reused (hotstart)':30}: {self.results['ngev_reused_in_hotstart']}"
        output += self.get_callback_counts_string(30)

        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':30}: {self.results['x']}"
        if optimal_gradient or all:
            output += f"\n\t{'Optimal obj. gradient':30}: {self.results['gradient']}"
        if optimal_constraints or all:
            output += f"\n\t{'Optimal constraints':30}: {self.results['constraints']}"
        if optimal_multipliers or all:
            output += f"\n\t{'Optimal multipliers (constr.)':30}: {self.results['multipliers']}"
        if optimal_jacobian or all:
            output += f"\n\t{'Optimal Jacobian':30}: {self.results['jacobian']}"

        output += '\n\t' + '-'*100
        print(output)
