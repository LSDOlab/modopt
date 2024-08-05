import numpy as np
from array_manager.api import DenseMatrix

from modopt import Optimizer
from modopt.utils.options_dictionary import OptionsDictionary


class SNOPTOptimizer(Optimizer):
    def initialize(self):
        self.solver_name = 'snopt'
        self.options.declare('solver_options', types=dict, default={})

        self.default_solver_options = {
            # Custom options for modopt
            'append2file': [False, False, bool],
            'continue_on_failure': [False, False, bool],
            # [Current value, default value, type]
            'Start type': ['Cold', 'Cold', str],  ##
            'Specs filename': [None, None, (str, type(None))],  ##
            'Print filename':
            ['SNOPT_print.out', 'SNOPT_print.out', str],  ##
            'Print frequency': [None, None, (int, type(None))],
            'Print level': [None, None,
                            (int, type(None))],  # minor print level
            'Summary': ['yes', 'yes', str],
            'Summary filename':
            ['SNOPT_summary.out', 'SNOPT_summary.out', str],
            'Summary frequency': [None, None, (int, type(None))],
            'Solution file': [None, None, (int, type(None))],
            'Solution filename': [
                'SNOPT_solution.out', 'SNOPT_solution.out',
                (str, type(None))
            ],
            'Solution print': [None, None, (bool, type(None))],
            'Major print level': [None, None, (int, type(None))],
            'Minor print level': [None, None, (int, type(None))],
            'Sticky parameters': [None, None, (int, type(None))],
            'Suppress': [None, None, (int, type(None))],
            'Time limit': [None, None, (float, type(None))],
            'Timing level': [None, None, (int, type(None))],
            'System information': [None, None, (int, type(None))],
            'Verify level': [None, None, (int, type(None))],
            'Max memory attempts': [10, 10, int],
            'Total character workspace':
            [None, None, (int, type(None))],
            'Total integer workspace': [None, None, (int, type(None))],
            'Total real workspace': [None, None, (int, type(None))],

            #'Problem minmax'        : ['Minimize','Minimize',str],
            'Proximal point': [None, None, (int, type(None))],
            #'QP solver'             : [None,None,str],    # Cholesky/CG/QN
            'Major feasibility': [None, None,
                                  (float, type(None))],  #tolCon
            'Major optimality': [None, None,
                                 (float, type(None))],  #tolOptNP
            'Minor feasibility': [None, None,
                                  (float, type(None))],  #tolx
            'Minor optimality': [None, None,
                                 (float, type(None))],  #tolOptQP
            'Minor phase1': [None, None,
                             (float, type(None))],  #tolOptFP
            'Feasibility tolerance': [None, None,
                                      (float, type(None))],  #tolx
            'Optimality tolerance': [None, None,
                                     (float, type(None))],  #tolOptQP
            'Iteration limit': [None, None, (int, type(None))],  #itnlim
            'Major iterations': [None, None,
                                 (int, type(None))],  #mMajor
            'Minor iterations': [None, None,
                                 (int, type(None))],  #mMinor
            'CG tolerance': [None, None, (float, type(None))],
            'CG preconditioning': [None, None, (int, type(None))],
            'CG iterations': [None, None, (int, type(None))],
            'Crash option': [None, None, (int, type(None))],
            'Crash tolerance': [None, None, (float, type(None))],
            'Debug level': [None, None, (int, type(None))],
            'Derivative level': [None, None, (int, type(None))],
            'Derivative linesearch': [None, None, (int, type(None))],
            'Derivative option': [None, None, (int, type(None))],
            'Elastic objective': [None, None, (int, type(None))],
            'Elastic mode': [None, None, (int, type(None))],
            'Elastic weight': [None, None, (float, type(None))],
            'Elastic weightmax': [None, None, (float, type(None))],
            'Hessian frequency': [None, None, (int, type(None))],
            'Hessian flush': [None, None, (int, type(None))],
            'Hessian type': [None, None, (int, type(None))],
            'Hessian updates': [None, None, (int, type(None))],
            'Infinite bound': [1.0e+20, 1.0e+20, float],
            'Major step limit': [None, None, (float, type(None))],
            'Unbounded objective': [None, None, (float, type(None))],
            'Unbounded step': [None, None, (float, type(None))],
            'Linesearch tolerance': [None, None, (float, type(None))],
            'Linesearch debug': [None, None, (int, type(None))],

            #'LU type'               : [None,None,str],   #partial/complete/rook
            'LU swap': [None, None, (float, type(None))],
            'LU factor tolerance': [None, None, (float, type(None))],
            'LU update tolerance': [None, None, (float, type(None))],
            'LU density': [None, None, (float, type(None))],
            'LU singularity': [None, None, (float, type(None))],
            'New superbasics': [None, None, (int, type(None))],
            'Partial pricing': [None, None, (int, type(None))],
            'Penalty parameter': [None, None, (float, type(None))],
            'Pivot tolerance': [None, None, (float, type(None))],
            'Reduced Hessian limit': [None, None, (int, type(None))],
            'Superbasics limit': [None, None, (int, type(None))],
            'Scale option': [None, None, (int, type(None))],
            'Scale tolerance': [None, None, (float, type(None))],
            'Scale print': [None, None, (int, type(None))],
            'Verbose': [True, True, bool]  ##
        }

        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[2], default=value[1])

        # Declare outputs
        self.available_outputs = {}
        self.options.declare('readable_outputs', values=([],), default=[])

        # Declare method specific options (implemented in the respective algorithm)
        self.declare_options()

        if hasattr(self.problem, "_compute_all"):
            if callable(self.problem._compute_all):
                self.compute_all = self.problem._compute_all

        self.x0 = self.problem.x0
        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.active_callbacks = ['obj', 'grad']

        if self.problem.nc > 0:
            # Uncomment the line below after testing our sqp_optimizer with atomics_lite
            # pC_px = DenseMatrix(self.problem.pC_px).numpy_array()
            self.con = self.problem._compute_constraints
            self.jac = self.problem._compute_constraint_jacobian
            self.active_callbacks += ['con', 'jac']
            # Uncomment the 2 lines below after testing our sqp_optimizer with atomics_lite
            # if pC_px.any() != 0:
            #     self.jac = lambda x: pC_px

    def declare_options(self):
        pass

    def update_SNOPT_options_object(self):
        try:
            from optimize import SNOPT_options
        except ImportError:
            raise ImportError("'SNOPT_options' from 'optimize' could not be imported. " \
                              "Make sure 'snopt-python' wrapper is correctly installed.")

        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])

        self.SNOPT_options_object = SNOPT_options()
        for key in self.solver_options:
            if key in ['append2file', 'continue_on_failure']:
                continue
            if 'filename' in key and hasattr(self, 'out_dir'):
                filename = self.solver_options[key]
                filepath = f"{self.out_dir}/{filename}" if filename else None
                self.SNOPT_options_object.setOption(key, filepath)
            else:
                self.SNOPT_options_object.setOption(key, self.solver_options[key])

    def setup_bounds(self):
        inf = self.solver_options['Infinite bound']
        xl = self.problem.x_lower
        xu = self.problem.x_upper
        self.x_lower = np.where(xl == -np.inf, -inf, xl)
        self.x_upper = np.where(xu ==  np.inf,  inf, xu)

    def setup_constraints(self, ):
        inf = self.solver_options['Infinite bound']
        cl = self.problem.c_lower
        cu = self.problem.c_upper
        self.c_lower = np.where(cl == -np.inf, -inf, cl)
        self.c_upper = np.where(cu ==  np.inf,  inf, cu)

    def print_results(self, 
                      optimal_variables=False,
                      optimal_constraints=False,
                      optimal_multipliers=False,
                      all=False):
        output  = "\n\tSolution from SNOPT:"
        output += "\n\t"+"-" * 100

        output += f"\n\t{'Problem':35}: {self.problem_name}"
        output += f"\n\t{'Solver':35}: {self.solver_name}"
        output += f"\n\t{'EXIT Code':35}: {self.results['info']}"
        output += f"\n\t{'Objective':35}: {self.results['objective']}"
        output += f"\n\t{'Num. of infeasible constraints':35}: {self.results['nInf']}"
        output += f"\n\t{'Sum. of infeasibilities':35}: {self.results['sInf']}"
        output += f"\n\t{'Num. of superbasic variables':35}: {self.results['nS']}"
        output += f"\n\t{'Major iterations':35}: {self.results['n_majiter']}"
        output += f"\n\t{'Total iterations':35}: {self.results['niter']}"
        output += f"\n\t{'Total time':35}: {self.total_time}"
        output += self.get_callback_counts_string(35)

        if optimal_variables or all:
            output += f"\n\t{'Optimal variables':35}: {self.results['x']}"
            output += f"\n\t{'Final states (variables)':35}: {self.results['x_states']}"
        if optimal_constraints or all:
            output += f"\n\t{'Optimal constraints':35}: {self.results['c']}"
            output += f"\n\t{'Final states (constraints)':35}: {self.results['c_states']}"
        if optimal_multipliers or all:
            output += f"\n\t{'Optimal multipliers (bounds)':35}: {self.results['lmult_x']}"
            output += f"\n\t{'Optimal multipliers (constr.)':35}: {self.results['lmult_c']}"

        output += '\n\t' + '-'*100

        print(output)
        pass