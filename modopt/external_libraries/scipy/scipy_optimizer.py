import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint
from scipy.optimize import BFGS, SR1

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

        # Method specific options are declared here
        self.declare_options()

        # fun : callable
        self.obj = self.problem.compute_objective
        # jac : callable
        # Only for CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
        if self.problem.compute_objective_gradient.__func__ is not Problem.compute_objective_gradient:
            self.grad = self.problem.compute_objective_gradient
        else:
            self.grad = self.options['gradient']

        # Only for COBYLA, SLSQP and trust-constr
        # Used to construct:
        # 1. a single object or a list of constraint() objects for trust-constr
        # 2. a list of dictionaries for COBYLA and SLSQP

        if self.prob_options['nc'] > 0:
            self.con = self.problem.compute_constraints
            if self.problem.compute_constraint_jacobian.__func__ is not Problem.compute_constraint_jacobian:
                self.jac = self.problem.compute_constraint_jacobian
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
        if self.problem.x_lower is None:
            self.bounds = None
            return None

        x_lower = self.problem.x_lower
        x_upper = self.problem.x_upper

        self.bounds = Bounds(x_lower, x_upper, keep_feasible=False)

    def setup_constraints(self, build_dict=True):
        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        # for SLSQP and COBYLA

        # self.constraints = []
        # self.constraints += []
        # self.constraints += []

        # # TODO: avoid loop of length nc
        # for i in range(self.options['nc']):
        #     con_dict = {}
        #     if self.problem.c_lower[i] == self.problem.c_upper[i]:
        #         con_dict['type'] = 'eq'
        #         def func(x):
        #             return self.con(x)[i] - self.problem.c_lower[i]

        #         def jac(x):
        #             return self.jac(x)[i]

        #     else:
        #         con_dict['type'] = 'ineq'

        if self.problem.c_lower is None:
            self.constraints = ()
            return None

        # To avoid can give as one eq and one ineq constraint
        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

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
                print('eq')

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
                    print('ineq1')

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

                    print('ineq2')

            # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
            # self.constraints = [
            #     con_dict_eq, con_dict_ineq1, con_dict_ineq2
            # ]

        else:
            # Adapt constraints a single/list of scipy LinearConstraint() or NonlinearConstraint() objects
            self.constraints = NonlinearConstraint(self.con,
                                                   c_lower,
                                                   c_upper,
                                                   jac=self.jac)

    # For callback, for every method except trust-constr
    # trust-constr can call with more information
    def save_xk(self, x):
        # Saving new x iterate on file
        name = self.problem_name
        nx = self.prob_options['nx']
        with open(name + '_x.out', 'a') as f:
            np.savetxt(f, x.reshape(1, nx))