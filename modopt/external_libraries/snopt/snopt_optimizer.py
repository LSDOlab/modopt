import numpy as np
from optimize import SNOPT_options

from array_manager.api import DenseMatrix

from modopt.api import Optimizer, Problem
from modopt.csdl_library import CSDLProblem


class SNOPTOptimizer(Optimizer):
    def initialize(self):
        self.solver_name = 'snopt_'
        self.options.declare('gradient',
                             default='exact',
                             values=['exact', 'fd'])
        self.options.declare('jacobian',
                             default='exact',
                             values=['exact', 'fd'])

        self.SNOPT_options = {
            # [Current value, default value, type]
            'Start type': ['Cold', 'Cold', str],  ##
            'Specs filename': [None, None, (str, type(None))],  ##
            'Print filename': ['SNOPT_print.out', 'SNOPT.out', str],  ##
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

        for option in self.SNOPT_options:
            # New option replaces spaces with underscores in option names
            new_option = option.replace(' ', '_')
            self.options.declare(
                new_option,
                default=self.SNOPT_options[option][1],
                types=self.SNOPT_options[option][2],
            )

        # Declare method specific options (implemented in the respective algorithm)
        self.declare_options()
        self.declare_outputs()

        self.obj = self.problem.objective
        # Restore back after teting sqp optzr. with atomics lite
        # self.x0 = self.problem.x.get_data()
        self.x0 = self.problem.x0

        # Exact gradient if provided else FD gradient
        if not isinstance(self.problem, CSDLProblem):
            if self.problem.compute_objective_gradient.__func__ is not Problem.compute_objective_gradient:
                self.grad = self.problem.objective_gradient
            else:
                self.grad = self.options['gradient']
        else:
            self.grad = self.problem.objective_gradient

        if self.problem.nc > 0:
            # Uncomment the line below after testing our sqp_optimizer with atomics_lite
            # pC_px = DenseMatrix(self.problem.pC_px).numpy_array()
            self.con = self.problem.constraints

            # Exact jac if provided else FD jac
            if not isinstance(self.problem, CSDLProblem):
                if self.problem.compute_constraint_jacobian.__func__ is not Problem.compute_constraint_jacobian:
                    self.jac = self.problem.constraint_jacobian

                # Uncomment the 2 lines below after testing our sqp_optimizer with atomics_lite
                # elif pC_px.any() != 0:
                #     self.jac = lambda x: pC_px
                else:
                    self.jac = self.options['jacobian']
            else:
                self.jac = self.problem.constraint_jacobian

    def update_SNOPT_options_object(self):
        self.SNOPT_options_object = SNOPT_options()
        for option in self.SNOPT_options:
            # New option replaces spaces with underscores in option names
            new_option = option.replace(' ', '_')
            self.SNOPT_options_object.setOption(
                option, self.options[new_option])

    def setup_bounds(self):
        inf = self.options['Infinite_bound']
        self.x_lower = np.where(self.problem.x_lower == -np.inf, -inf,
                                self.problem.x_lower)
        self.x_upper = np.where(self.problem.x_upper == np.inf, inf,
                                self.problem.x_upper)

    def setup_constraints(self, ):

        if self.problem.c_lower.size == 0 and self.problem.c_upper.size:
            return None

        inf = self.options['Infinite_bound']
        self.c_lower = np.where(self.problem.c_lower == -np.inf, -inf,
                                self.problem.c_lower)
        self.c_upper = np.where(self.problem.c_upper == np.inf, inf,
                                self.problem.c_upper)

    # For callback, for every method
    # Overrides base class update_outputs()
    # def update_outputs(self, xk):
    #     name = self.problem_name
    #     with open(name + '_x.out', 'a') as f:
    #         np.savetxt(f, xk.reshape(1, xk.size))

    #     self.outputs['x'] = np.append(
    #         self.outputs['x'],
    #         #   xk.reshape((1, ) + (xk.size,)),
    #         xk.reshape((1, ) + xk.shape),
    #         axis=0)

    # def save_xk(self, x):
    #     # Saving new x iterate on file
    #     name = self.problem_name
    #     nx = self.problem.nx

    #     with open(name + '_x.out', 'a') as f:
    #         np.savetxt(f, x.reshape(1, nx))

    # print_results for scipy_library overrides print_results from Optimizer()
    # summary table and compact print does not work
    def print_results(self, **kwargs):
        # TODO: Use pthon file snopt.py and mics.py and result() object to print in modopt format
        # Testing to verify the design variable data
        # print(np.loadtxt(self.problem_name+'_x.out') - self.outputs['x_array'])
        # print("\n", "\t" * 1, "==============")
        # print("\t" * 1, "SNOPT summary:")
        # print("\t" * 1, "==============", "\n")
        # print("\t" * 1, "Problem", "\t" * 3, ':', self.problem_name)
        # print("\t" * 1, "Solver", "\t" * 3, ':', self.solver_name)

        # print(self.snopt_output)
        pass