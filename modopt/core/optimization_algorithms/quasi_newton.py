import numpy as np
import time

from modopt.api import Optimizer
from modopt.line_search_algorithms import ScipyLS
# from modopt.approximate_hessians import BFGS
from modopt.approximate_hessians import BFGSM1 as BFGS


class QuasiNewton(Optimizer):
    def initialize(self):
        self.solver_name = 'bfgs'

        self.obj = self.problem.compute_objective
        self.grad = self.problem.compute_objective_gradient

        self.options.declare('opt_tol', types=float)

        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
            'num_f_evals': int,
            'num_g_evals': int,
            'step': float,
        }
        self.options.declare('outputs',
                             types=list,
                             default=[
                                 'itr', 'obj', 'x', 'opt', 'time',
                                 'num_f_evals', 'num_g_evals', 'step'
                             ])

    def setup(self):
        self.LS = ScipyLS(f=self.obj, g=self.grad)
        self.QN = BFGS(nx=self.problem.nx)

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        # x0 = self.prob_options['x0']
        x0 = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        max_itr = self.options['max_itr']

        obj = self.obj
        grad = self.grad

        LS = self.LS
        QN = self.QN

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        f_k = self.obj(x_k)
        g_k = self.grad(x_k)

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

            # Hessian approximation
            B_k = QN.B_k

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            p_k = np.linalg.solve(B_k, -g_k)

            # Compute the step length along the search direction via a line search
            alpha, f_k, g_new, slope_new, new_f_evals, new_g_evals, converged = LS.search(
                x=x_k, p=p_k, f0=f_k, g0=g_k)

            num_f_evals += new_f_evals
            num_g_evals += new_g_evals

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                d_k = p_k * 1e-4

                x_k += d_k
                f_k = obj(x_k)

                g_new = grad(x_k)
                w_k = g_new - g_k
                g_k = g_new

            else:
                d_k = alpha * p_k
                x_k += d_k

                if g_new == 'Unavailable':
                    g_new = grad(x_k)
                w_k = g_new - g_k
                g_k = g_new

            opt = np.linalg.norm(g_k)

            # Update the Hessian approximation
            QN.update(d_k, w_k)

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
