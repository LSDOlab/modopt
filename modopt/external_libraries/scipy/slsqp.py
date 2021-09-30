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

    def setup(self):
        # Adapt bounds as scipy Bounds() object
        self.setup_bounds()

        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        self.setup_constraints(build_dict=True)

    def solve(self):
        # Assign shorter names to variables and methods
        method = 'SLSQP'
        # nx = self.prob_options['nx']
        # nc = self.options['nc']

        # x0 = self.prob_options['x0']
        x0 = self.problem.x.get_data()

        ftol = self.options['ftol']
        # opt_tol = self.options['opt_tol']
        # feas_tol = self.options['feas_tol']

        maxiter = self.options['maxiter']
        eps = self.options['eps']
        finite_diff_rel_step = self.options['finite_diff_rel_step']

        obj = self.obj
        grad = self.grad
        # Note: SLSQP does not take hessians

        bounds = self.bounds
        constraints = self.constraints  # (contains eq,ineq constraints and jacobian)
        callback = self.save_xk
        disp = self.options['disp']
        # con = self.con
        # jac = self.jac

        start_time = time.time()

        # Call SLSQP algorithm from scipy (options are specific to SLSQP)
        # Note: f_tol is the precision tolerance for the objective(not the same as opt_tol)

        print(obj(x0))

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
            # callback=callback,
            callback=None,
            options={
                #   'func': obj,
                'maxiter': maxiter,
                'ftol': ftol,
                'iprint': 1,
                'disp': disp,
                'return_all': True,
                'eps': eps,
                'finite_diff_rel_step': finite_diff_rel_step
            })

        print(result)
