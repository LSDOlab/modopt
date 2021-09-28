import numpy as np
from scipy.optimize import minimize, Bounds
import time

from .scipy_optimizer import ScipyOptimizer


class COBYLA(ScipyOptimizer):
    def declare_options(self):
        self.solver_name += 'cobyla'

        # Solver-specific options exactly as in scipy with defaults
        self.options.declare('maxiter', default = 1000, types=int)
        self.options.declare('disp', default = True, types=bool)
        self.options.declare('rhobeg', default = 1.0, types=float)
        # Objective precision
        self.options.declare('tol', default = None, types=(float, type(None)))
        # Constraint violation
        self.options.declare('catol', default = 0.0002, types=float)

    def setup(self):
        # Adapt bounds as scipy Bounds() object
        self.setup_bounds()
        
        # Adapt constraints as a list of dictionaries with constraints = 0 or >= 0
        self.setup_constraints(build_dict=True)

    def solve(self):
        # Assign shorter names to variables and methods
        method = 'COBYLA'
        # nx = self.prob_options['nx']
        # nc = self.options['nc']

        x0 = self.prob_options['x0']
        tol = self.options['tol']
        catol = self.options['catol']
        # opt_tol = self.options['opt_tol']
        # feas_tol = self.options['feas_tol']
        
        maxiter = self.options['maxiter']
        rhobeg = self.options['rhobeg']

        obj = self.obj
        grad = self.grad
        # Note: SLSQP does not take hessians

        bounds= self.bounds
        
        constraints= self.constraints # (contains eq,ineq constraints and jacobian)
        
        # COBYLA does not support callback
        callback = None
        disp = self.options['disp']
        # con = self.con
        # jac = self.jac

        start_time = time.time()

        # Call SLSQP algorithm from scipy (options are specific to SLSQP)
        # Note: f_tol is the precision tolerance for the objective(not the same as opt_tol)
        # COBYLA has no return_all option
        result = minimize(obj, x0, args=(),  method=method, jac=None, 
        hess=None, hessp=None, bounds=bounds, constraints=constraints, 
        tol=tol, callback=callback, 
        options={'maxiter': maxiter, 
        'catol': catol, 'disp': disp, 'rhobeg': rhobeg})

        print(result)
        
