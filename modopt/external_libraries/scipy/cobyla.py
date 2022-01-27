import numpy as np
from scipy.optimize import minimize, Bounds, show_options
import time

from .scipy_optimizer import ScipyOptimizer


class COBYLA(ScipyOptimizer):
    def declare_options(self):
        self.solver_name += 'cobyla'

        # Solver-specific options exactly as in scipy with defaults
        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('disp', default=False, types=bool)
        self.options.declare('rhobeg', default=1.0, types=float)
        # Objective precision
        self.options.declare('tol',
                             default=None,
                             types=(float, type(None)))
        # Constraint violation
        self.options.declare('catol', default=0.0002, types=float)

    def declare_outputs(self, ):
        self.default_outputs_format = {}

        self.options.declare('outputs', types=list, default=[])

    def setup(self):
        # Adapt constraints with bounds as a list of dictionaries with constraints = 0 or >= 0
        self.setup_constraints()

    def setup_constraints(self, ):

        c_lower = self.problem.c_lower
        c_upper = self.problem.c_upper

        x_lower = self.problem.x_lower
        x_upper = self.problem.x_upper

        # Adapt bounds as ineq constraints
        self.setup_bounds()

        if c_lower is None and self.bounds is None:
            self.constraints = ()
            return None

        self.constraints = []

        # Adapt eq/ineq constraints and bounds as a list of dictionaries with constraints >= 0 for COBYLA
        # Note: COBYLA only supports inequality constraints

        if self.bounds is not None:
            # Remove constraints with -np.inf as lower bound
            lower_bound_indices = np.where(x_lower != -np.inf)[0]
            # Remove constraints with np.inf as upper bound
            upper_bound_indices = np.where(x_upper != np.inf)[0]

            if len(lower_bound_indices) > 0:
                con_dict_bound1 = {}
                con_dict_bound1['type'] = 'ineq'

                def func_bound1(x):
                    return x[lower_bound_indices] - x_lower[
                        lower_bound_indices]

                con_dict_bound1['fun'] = con_dict_bound1

                self.constraints.append(con_dict_bound1)

            if len(upper_bound_indices) > 0:
                con_dict_bound2 = {}
                con_dict_bound2['type'] = 'ineq'

                def func_bound2(x):
                    return x_upper[upper_bound_indices] - x[
                        upper_bound_indices]

                con_dict_bound1['fun'] = con_dict_bound2

                self.constraints.append(con_dict_bound2)

        if c_lower is not None:
            # Remove constraints with -np.inf as lower bound
            lower_c_indices = np.where(c_lower != -np.inf)[0]
            # Remove constraints with np.inf as upper bound
            upper_c_indices = np.where(c_upper != np.inf)[0]

            if len(lower_c_indices) > 0:
                con_dict_ineq1 = {}
                con_dict_ineq1['type'] = 'ineq'

                def func_ineq1(x):
                    return self.con(
                        x)[lower_c_indices] - c_lower[lower_c_indices]

                con_dict_ineq1['fun'] = func_ineq1

                self.constraints.append(con_dict_ineq1)

            if len(upper_c_indices) > 0:
                con_dict_ineq2 = {}
                con_dict_ineq2['type'] = 'ineq'

                def func_ineq2(x):
                    return c_upper[upper_c_indices] - self.con(
                        x)[upper_c_indices]

                con_dict_ineq2['fun'] = func_ineq2

                self.constraints.append(con_dict_ineq2)

    def solve(self):
        # Assign shorter names to variables and methods
        method = 'COBYLA'

        x0 = self.x0

        tol = self.options['tol']
        catol = self.options['catol']

        maxiter = self.options['maxiter']
        rhobeg = self.options['rhobeg']

        obj = self.obj
        grad = self.grad

        bounds = self.bounds

        constraints = self.constraints  # (contains eq,ineq constraints and jacobian)

        # COBYLA does not support callback
        # callback = None
        disp = self.options['disp']

        start_time = time.time()

        # COBYLA has no return_all option
        result = minimize(obj,
                          x0,
                          args=(),
                          method=method,
                          jac=None,
                          hess=None,
                          hessp=None,
                          bounds=bounds,
                          constraints=constraints,
                          tol=None,
                          callback=None,
                          options={
                              'maxiter': maxiter,
                              'tol': tol,
                              'catol': catol,
                              'disp': disp,
                              'rhobeg': rhobeg
                          })

        # print(result)

        end_time = time.time()
        self.total_time = end_time - start_time

        self.scipy_output = result

        # show_options(solver='minimize', method=None, disp=True)