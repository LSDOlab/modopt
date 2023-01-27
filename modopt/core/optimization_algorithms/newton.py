import numpy as np
import time

from modopt.api import Optimizer
from modopt.line_search_algorithms import ScipyLS


class Newton(Optimizer):
    def initialize(self):
        self.solver_name = 'newton'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.hess = self.problem._compute_objective_hessian

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-7, types=float)

        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
            'num_f_evals': int,
            'num_g_evals': int,
            'step': float,
        }

        # self.default_outputs_format = {
        #     'itr': ('Iteration number', int),
        #     'obj': ('Objective values', float),
        #     # for arrays from each iteration, shapes need to be declared
        #     'x': ('Design variable values', float, (self.problem.nx, )),
        #     'opt': ('Optimality', float),
        #     'time': ('Time', float),
        #     'num_f_evals': ('Number of objective evaluations', int),
        #     'num_g_evals': ('Number of gradient evaluations', int),
        #     'step': ('Step lengths from line search', float),
        # }

        self.options.declare('outputs',
                             types=list,
                             default=[
                                 'itr', 'obj', 'x', 'opt', 'time',
                                 'num_f_evals', 'num_g_evals', 'step'
                             ])

    def setup(self):
        self.LS = ScipyLS(f=self.obj, g=self.grad)

    def solve(self):

        # Assign shorter names to variables and methods
        nx = self.problem.nx
        # x0 = self.prob_options['x0']
        x0 = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        max_itr = self.options['max_itr']

        obj = self.obj
        grad = self.grad
        hess = self.hess

        LS = self.LS

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        f_k = self.obj(x_k)
        g_k = self.grad(x_k)
        H_k = self.hess(x_k)

        # Iteration counter
        itr = 0

        opt = np.linalg.norm(g_k)
        num_f_evals = 1
        num_g_evals = 1

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time,
                            num_f_evals=num_f_evals,
                            num_g_evals=num_g_evals,
                            step=0.)

        while (opt > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            p_k = np.linalg.solve(H_k, -g_k)

            # Compute the step length along the search direction via a line search
            alpha, f_k, g_k, slope_new, new_f_evals, new_g_evals, converged = LS.search(
                x=x_k, p=p_k, f0=f_k, g0=g_k)

            num_f_evals += new_f_evals
            num_g_evals += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                alpha = 1e-4
                x_k += p_k * alpha
                f_k = obj(x_k)
                g_k = grad(x_k)

            else:
                x_k += alpha * p_k
                if g_k == 'Unavailable':
                    g_k = grad(x_k)

            H_k = hess(x_k)

            opt = np.linalg.norm(g_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time,
                                num_f_evals=num_f_evals,
                                num_g_evals=num_g_evals,
                                step=alpha)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time