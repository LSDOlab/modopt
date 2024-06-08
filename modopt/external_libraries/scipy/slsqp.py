import numpy as np
from scipy.optimize import minimize, Bounds
import time

from .scipy_optimizer import ScipyOptimizer


class SLSQP(ScipyOptimizer):
    def declare_options(self):
        self.solver_name += 'slsqp'

        # Solver-specific options exactly as in scipy with defaults
        self.options.declare('maxiter', default=100, types=int)
        self.options.declare('disp', default=False, types=bool)
        self.options.declare('eps',
                             default=1.4901161193847656e-08,
                             types=float)
        self.options.declare('finite_diff_rel_step',
                             default=None,
                             types=(type(None), np.ndarray))
        # Objective precision
        self.options.declare('ftol', default=1e-6, types=float)

    def declare_outputs(self, ):
        self.available_outputs = {
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
        }

        self.options.declare('outputs', values=([],['x']), default=[])

    def setup(self):
        # Adapt bounds as scipy Bounds() object
        self.setup_bounds()

        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        self.setup_constraints()

    def setup_constraints(self, ):

        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

        # unconstrained problem: if c_lower=None 
        # (in Problem(), OM, csdl, cutest problems) 
        if not self.problem.constrained:
            # print('No constraints')
            self.constraints = ()
            return None

        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0 for SLSQP
        eq_indices = np.where(c_upper == c_lower)[0]
        ineq_indices = np.where(c_upper != c_lower)[0]

        self.constraints = []
        if len(eq_indices) > 0:
            con_dict_eq = {}
            con_dict_eq['type'] = 'eq'

            def func_eq(x):
                # print('EQ:')
                # print(self.con(x)[eq_indices])
                # print(c_lower[eq_indices])

                return self.con(x)[eq_indices] - c_lower[eq_indices]

            con_dict_eq['fun'] = func_eq

            if type(self.jac) != str:

                def jac_eq(x):
                    return self.jac(x)[eq_indices]

                con_dict_eq['jac'] = jac_eq

            self.constraints.append(con_dict_eq)

        if len(ineq_indices) > 0:
            # Remove constraints with -np.inf as lower bound
            c_lower_ineq = c_lower[ineq_indices]
            lower_ineq_indices = ineq_indices[np.where(
                c_lower_ineq != -np.inf)[0]]

            if len(lower_ineq_indices) > 0:
                con_dict_ineq1 = {}
                con_dict_ineq1['type'] = 'ineq'

                def func_ineq1(x):
                    # print('INEQ1:')
                    # print(self.con(x)[lower_ineq_indices])
                    # print(c_lower[lower_ineq_indices])

                    return self.con(x)[lower_ineq_indices] - c_lower[
                        lower_ineq_indices]

                con_dict_ineq1['fun'] = func_ineq1

                if type(self.jac) != str:

                    def jac_ineq1(x):
                        return self.jac(x)[lower_ineq_indices]

                    con_dict_ineq1['jac'] = jac_ineq1

                self.constraints.append(con_dict_ineq1)

            # Remove constraints with np.inf as upper bound
            c_upper_ineq = c_upper[ineq_indices]
            upper_ineq_indices = ineq_indices[np.where(
                c_upper_ineq != np.inf)[0]]

            if len(upper_ineq_indices) > 0:
                con_dict_ineq2 = {}
                con_dict_ineq2['type'] = 'ineq'

                def func_ineq2(x):
                    # print('INEQ2:')
                    # print(self.con(x)[upper_ineq_indices])
                    # print(c_upper[upper_ineq_indices])
                    return c_upper[upper_ineq_indices] - self.con(
                        x)[upper_ineq_indices]

                con_dict_ineq2['fun'] = func_ineq2

                if type(self.jac) != str:

                    def jac_ineq2(x):
                        return -self.jac(x)[upper_ineq_indices]

                    con_dict_ineq2['jac'] = jac_ineq2

                self.constraints.append(con_dict_ineq2)

    def solve(self):
        # Assign shorter names to variables and methods
        method = 'SLSQP'

        x0 = self.problem.x0 * 1.

        self.update_outputs(x0)

        ftol = self.options['ftol']

        maxiter = self.options['maxiter']
        eps = self.options['eps']
        finite_diff_rel_step = self.options['finite_diff_rel_step']

        obj = self.obj
        grad = self.grad
        # Note: SLSQP does not take Hessians

        bounds = self.bounds
        constraints = self.constraints  # (contains eq,ineq constraints and jacobian)
        callback = self.update_outputs

        disp = self.options['disp']

        start_time = time.time()

        # Call SLSQP algorithm from scipy (options are specific to SLSQP)
        # Note: f_tol is the precision tolerance for the objective(not the same as opt_tol)

        result = minimize(
            obj,
            x0,
            args=(),
            method=method,
            jac=grad,
            hess=None,
            hessp=None,
            bounds=bounds,
            constraints=constraints,
            tol=None,
            callback=callback,
            options={
                #   'func': obj,
                'maxiter': maxiter,
                'ftol': ftol,
                'iprint': 1,
                'disp': disp,
                'eps': eps,
            })

        end_time = time.time()
        self.total_time = end_time - start_time

        self.scipy_output = result
