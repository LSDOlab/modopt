import numpy as np
from scipy.optimize import minimize, Bounds
import time

from .scipy_optimizer import ScipyOptimizer


class BFGS(ScipyOptimizer):
    def declare_options(self):
        self.solver_name += 'bfgs'

        # Solver-specific options exactly as in scipy with defaults
        self.options.declare('maxiter',
                             default=None,
                             types=(int, type(None)))
        self.options.declare('disp', default=True, types=bool)
        self.options.declare('eps',
                             default=1.4901161193847656e-08,
                             types=float)
        self.options.declare('finite_diff_rel_step',
                             default=None,
                             types=(type(None), np.ndarray))
        self.options.declare('return_all', default=True, types=bool)
        # Gradient norm
        self.options.declare('gtol', default=1e-5, types=float)
        # Order of gradient norm (inf is max, -inf is min)
        self.options.declare('norm', default=np.inf, types=float)

    def setup(self):
        # start a new file for saving callback() save_xk()
        pass

    def solve(self):
        # Assign shorter names to variables and methods
        method = 'BFGS'
        # nx = self.prob_options['nx']
        # nc = self.options['nc']

        x0 = self.prob_options['x0']
        gtol = self.options['gtol']
        norm = self.options['norm']
        eps = self.options['eps']
        finite_diff_rel_step = self.options['finite_diff_rel_step']

        # opt_tol = self.options['opt_tol']
        # feas_tol = self.options['feas_tol']

        maxiter = self.options['maxiter']
        disp = self.options['disp']
        return_all = self.options['return_all']

        obj = self.obj
        grad = self.grad

        # COBYLA does not support callback
        callback = self.save_xk
        disp = self.options['disp']
        # con = self.con
        # jac = self.jac

        start_time = time.time()

        # Call SLSQP algorithm from scipy (options are specific to SLSQP)
        # Note: f_tol is the precision tolerance for the objective(not the same as opt_tol)

        # TODO: Make sure 'finite_diff_rel_step': None' is an option for BFGS
        # TODO: What does return_all=True do, where does it return
        #  a list of the best solution at each of the iterations? ANS: inside result
        result = minimize(obj,
                          x0,
                          args=(),
                          method=method,
                          jac=grad,
                          hess=None,
                          hessp=None,
                          bounds=None,
                          constraints=None,
                          tol=None,
                          callback=callback,
                          options={
                              'gtol': gtol,
                              'norm': norm,
                              'eps': eps,
                              'maxiter': maxiter,
                              'disp': disp,
                              'return_all': return_all,
                          })
        #  'finite_diff_rel_step': None})

        print(result)
