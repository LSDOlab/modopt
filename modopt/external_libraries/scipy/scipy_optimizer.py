import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS, SR1
from array_manager.api import DenseMatrix

from modopt.api import Optimizer, Problem


class ScipyOptimizer(Optimizer):
    def initialize(self):
        self.solver_name = 'scipy_'
        self.options.declare('gradient',
                             default='2-point',
                             values=['2-point', '3-point', 'cs'])
        self.options.declare('jacobian',
                             default='2-point',
                             values=['2-point', '3-point', 'cs'])
        self.options.declare(
            'hessian',
            default='2-point',
            values=['2-point', '3-point', 'cs', 'bfgs', 'sr1'])

        self.options.declare(
            'lagrangian_hessian',
            default='2-point',
            values=['2-point', '3-point', 'cs', 'bfgs', 'sr1'])

        # Declare method specific options (implemented in the respective algorithm)
        self.declare_options()
        self.declare_outputs()

        self.obj = self.problem.compute_objective
        # Restore back after teting sqp optzr. with atomics lite
        # self.x0 = self.problem.x.get_data()
        self.x0 = self.problem.x0

        # Gradient only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg,
        # trust-krylov, trust-exact and trust-constr
        if self.problem.compute_objective_gradient.__func__ is not Problem.compute_objective_gradient:
            self.grad = self.problem.compute_objective_gradient
        else:
            self.grad = self.options['gradient']

        # Only for COBYLA, SLSQP and trust-constr
        # Used to construct:
        # 1. a single object or a list of constraint() objects for trust-constr
        # 2. a list of dictionaries for COBYLA and SLSQP

        if self.problem.nc > 0:
            # Uncomment the line below after testing our sqp_optimizer with atomics_lite
            # pC_px = DenseMatrix(self.problem.pC_px).numpy_array()
            self.con = self.problem.compute_constraints
            if self.problem.compute_constraint_jacobian.__func__ is not Problem.compute_constraint_jacobian:
                self.jac = self.problem.compute_constraint_jacobian

            # Uncomment the 2 lines below after testing our sqp_optimizer with atomics_lite
            elif pC_px.any() != 0:
                self.jac = lambda x: pC_px
            else:
                self.jac = self.options['jacobian']

        # SETUP OBJECTIVE HESSIAN OR HVP

        if self.problem.compute_objective_hessian.__func__ is not Problem.compute_objective_hessian:
            self.hess = self.problem.compute_objective_hessian
        elif self.options['hessian'] == 'BFGS':
            # Users will have to manually modify this if needed
            self.hess = BFGS(exception_strategy='skip_update',
                             min_curvature=None,
                             init_scale='auto')
        elif self.options['hessian'] == 'SR1':
            # Users will have to manually modify this if needed
            self.hess = SR1(min_denominator=1e-08, init_scale='auto')

        # Only for Newton-CG, trust-ncg, trust-krylov, trust-constr.
        # Note: This is always Objective hessian even for trust-constr (Constraint hessians defined in constraints)
        elif self.problem.compute_objective_hvp.__func__ is not Problem.compute_objective_hvp:
            self.hvp = self.problem.compute_objective_hvp
        else:
            self.hess = self.options['hessian']

        # SETUP LAGRANGIAN HESSIAN (only for trust-constr)
        if self.problem.compute_lagrangian_hessian.__func__ is not Problem.compute_lagrangian_hessian:
            self.lag_hess = self.problem.compute_lagrangian_hessian

        elif self.options['lagrangian_hessian'] == 'BFGS':
            # Users will have to manually modify this if needed
            self.lag_hess = BFGS(exception_strategy='skip_update',
                                 min_curvature=None,
                                 init_scale='auto')
        elif self.options['lagrangian_hessian'] == 'SR1':
            # Users will have to manually modify this if needed
            self.lag_hess = SR1(min_denominator=1e-08,
                                init_scale='auto')

        else:
            self.lag_hess = self.options['lagrangian_hessian']

    def setup_bounds(self):
        # Adapt bounds as scipy Bounds() object
        # Only for Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods

        # TODO: check for bugs for the if condition from 2 lines below
        # if self.problem.x_lower.all(
        # ) != -np.inf and self.problem.x_upper.all() != np.inf:
        if self.problem.x_lower.any(
        ) != -np.inf or self.problem.x_upper.any() != np.inf:
            self.bounds = None
            return None

        x_lower = self.problem.x_lower
        x_upper = self.problem.x_upper

        self.bounds = Bounds(x_lower, x_upper, keep_feasible=False)

    def setup_constraints(self, build_dict=True):

        # To avoid can give as one eq and one ineq constraint
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

        if c_lower.size == 0:
            self.constraints = ()
            return None

        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        # For SLSQP and COBYLA
        # Note: COBYLA only supports inequality constraints

        if build_dict:
            eq_indices = np.where(c_upper == c_lower)[0]
            ineq_indices = np.where(c_upper != c_lower)[0]

            self.constraints = []
            if len(eq_indices) > 0:
                con_dict_eq = {}
                con_dict_eq['type'] = 'eq'

                def func_eq(x):
                    return self.con(x)[
                        eq_indices] - self.problem.c_lower[eq_indices]

                con_dict_eq['fun'] = func_eq

                if type(self.jac) != str:

                    def jac_eq(x):
                        return self.jac(x)[eq_indices]

                    con_dict_eq['jac'] = jac_eq

                self.constraints.append(con_dict_eq)
                # print('eq')

            if len(ineq_indices) > 0:

                # Remove -np.inf constraints with -np.inf as lower bound
                c_lower_ineq = c_lower[ineq_indices]
                lower_ineq_indices = ineq_indices[np.where(
                    c_lower_ineq != -np.inf)[0]]

                if len(lower_ineq_indices) > 0:
                    con_dict_ineq1 = {}
                    con_dict_ineq1['type'] = 'ineq'

                    def func_ineq1(x):
                        return self.con(x)[
                            lower_ineq_indices] - self.problem.c_lower[
                                lower_ineq_indices]

                    con_dict_ineq1['fun'] = func_ineq1

                    if type(self.jac) != str:

                        def jac_ineq1(x):
                            return self.jac(x)[lower_ineq_indices]

                        con_dict_ineq1['jac'] = jac_ineq1

                    self.constraints.append(con_dict_ineq1)
                    # print('ineq1')

                # Remove np.inf constraints with -np.inf as lower bound
                c_upper_ineq = c_upper[ineq_indices]
                upper_ineq_indices = ineq_indices[np.where(
                    c_upper_ineq != np.inf)[0]]

                if len(upper_ineq_indices) > 0:
                    con_dict_ineq2 = {}
                    con_dict_ineq2['type'] = 'ineq'

                    def func_ineq2(x):
                        return self.problem.c_upper[
                            upper_ineq_indices] - self.con(
                                x)[upper_ineq_indices]

                    con_dict_ineq2['fun'] = func_ineq2

                    if type(self.jac) != str:

                        def jac_ineq2(x):
                            print('jac:', self.jac)
                            return -self.jac(x)[upper_ineq_indices]

                        con_dict_ineq2['jac'] = jac_ineq2

                    self.constraints.append(con_dict_ineq2)

                    # print('ineq2')

        else:
            # Adapt constraints a single/list of scipy LinearConstraint() or NonlinearConstraint() objects
            self.constraints = NonlinearConstraint(self.con,
                                                   c_lower,
                                                   c_upper,
                                                   jac=self.jac)

    # For callback, for every method except trust-constr
    # trust-constr can call with more information
    # Overrides base class update_outputs()
    def update_outputs(self, xk):
        name = self.problem_name
        with open(name + '_x.out', 'a') as f:
            np.savetxt(f, xk.reshape(1, xk.size))

        self.outputs['x'] = np.append(
            self.outputs['x'],
            #   xk.reshape((1, ) + (xk.size,)),
            xk.reshape((1, ) + xk.shape),
            axis=0)

    def save_xk(self, x):
        # Saving new x iterate on file
        name = self.problem_name
        nx = self.problem.nx

        with open(name + '_x.out', 'a') as f:
            np.savetxt(f, x.reshape(1, nx))

    # print_results for scipy_library overrides print_results from Optimizer()
    # summary table and compact print does not work
    def print_results(self, **kwargs):
        # Testing to verify the design variable data
        # print(np.loadtxt(self.problem_name+'_x.out') - self.outputs['x_array'])
        print("\n", "\t" * 1, "==============")
        print("\t" * 1, "Scipy summary:")
        print("\t" * 1, "==============", "\n")
        print("\t" * 1, "Problem", "\t" * 3, ':', self.problem_name)
        print("\t" * 1, "Solver", "\t" * 3, ':', self.solver_name)
        print("\t" * 1, "Success", "\t" * 3, ':',
              self.scipy_output['success'])
        print("\t" * 1, "Message", "\t" * 3, ':',
              self.scipy_output['message'])
        print("\t" * 1, "Objective", "\t" * 3, ':',
              self.scipy_output['fun'])
        if 'njev' in self.scipy_output:
            print("\t" * 1, "Gradient norm", "\t" * 3, ':',
                  np.linalg.norm(self.scipy_output['jac']))

        print("\t" * 1, "Total time", "\t" * 3, ':', self.total_time)
        if 'nit' in self.scipy_output:
            print("\t" * 1, "Major iterations", "\t" * 2, ':',
                  self.scipy_output['nit'])

        # if self.scipy_output['nfev'] is not None:
        print("\t" * 1, "Total function evaluations", "\t" * 1, ':',
              self.scipy_output['nfev'])
        if 'njev' in self.scipy_output:
            print("\t" * 1, "Total gradient evaluations", "\t" * 1, ':',
                  self.scipy_output['njev'])

        allowed_keys = {
            'optimal_variables',
            # 'summary_table',
            # 'compact_print'
        }
        self.__dict__.update((key, False) for key in allowed_keys)
        self.__dict__.update((key, val) for key, val in kwargs.items()
                             if key in allowed_keys)

        if self.optimal_variables:
            print("\t" * 1, "Optimal variables", "\t" * 2, ':',
                  self.scipy_output['x'])

        print("\t", "===========================================")