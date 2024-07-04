import numpy as np
from scipy.optimize import minimize, Bounds
import time
from modopt.utils.options_dictionary import OptionsDictionary

from .scipy_optimizer import ScipyOptimizer


class SLSQP(ScipyOptimizer):
    def declare_options(self):
        self.solver_name += 'slsqp'
        self.options.declare('solver_options', types=dict, default={})
        self.default_solver_options = {
            'maxiter': (int, 100),
            'ftol': (float, 1e-6),
            'disp': (bool, False),
            'callback': ((type(None), callable), None),
        }
        # Used for verifying the keys and value-types of user-provided solver_options, 
        # and generating an updated pure Python dictionary to provide pyslsqp.optimize()
        self.solver_options = OptionsDictionary()
        for key, value in self.default_solver_options.items():
            self.solver_options.declare(key, types=value[0], default=value[1])

    def declare_outputs(self, ):
        self.available_outputs = {'x': (float, (self.problem.nx,))}
        self.options.declare('outputs', values=([],['x']), default=[])

    def setup(self):
        self.setup_outputs()
        # Check if user-provided solver_options have valid keys and value-types
        self.solver_options.update(self.options['solver_options'])

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
        method = 'SLSQP'
        solver_options = self.solver_options.get_pure_dict()
        user_callback = solver_options.pop('callback', None)

        def callback(x): 
            self.update_outputs(x=x)
            if user_callback: user_callback(x) 

        x0 = self.problem.x0 * 1.
        self.update_outputs(x=x0)

        obj = self.obj
        grad = self.grad
        bounds = self.bounds
        constraints = self.constraints  # (contains eq,ineq constraints and jacobian)

        # Call SLSQP algorithm from scipy (options are specific to SLSQP)
        start_time = time.time()
        self.results = minimize(
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
            options=solver_options
            )
        self.total_time = time.time() - start_time
        
        return self.results
