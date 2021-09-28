import numpy as np
import time

from modopt.api import Optimizer
from modopt.line_search_algorithms import ScipyLS


class Newton(Optimizer):
    def initialize(self):
        self.solver_name = 'newton'

        self.obj = self.problem.compute_objective
        self.grad = self.problem.compute_objective_gradient
        self.hess = self.problem.compute_objective_hessian

        self.options.declare('opt_tol', types=float)

    def setup(self):
        self.LS = ScipyLS(f=self.obj, g=self.grad)

    def solve(self):

        # Assign shorter names to variables and methods
        nx = self.prob_options['nx']
        x0 = self.prob_options['x0']
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

        # Initialize output arrays
        itr_array = np.array([
            0,
        ])
        x_array = x0.reshape(1, nx)
        obj_array = np.array([f_k * 1.])
        opt_array = np.array([np.linalg.norm(g_k)])
        num_f_evals_array = np.array([1])
        num_g_evals_array = np.array([1])
        step_array = np.array([
            0.,
        ])
        time_array = np.array([time.time() - start_time])

        while (opt_array[-1] > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate
            p_k = np.linalg.solve(H_k, -g_k)

            # Compute the step length along the search direction via a line search
            alpha, f_k, g_k, slope_new, num_f_evals, num_g_evals, converged = LS.search(
                x=x_k, p=p_k, f0=f_k, g0=g_k)

            # A step of length 1e-4 is taken along p_k if line search does not converge
            if not converged:
                x_k += p_k * 1e-4
                f_k = obj(x_k)
                g_k = grad(x_k)

            else:
                x_k += alpha * p_k
                if g_k == 'Unavailable':
                    g_k = grad(x_k)

            H_k = hess(x_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Append output arrays with new values from the current iteration
            itr_array = np.append(itr_array, itr)
            x_array = np.append(x_array, x_k.reshape(1, nx), axis=0)
            obj_array = np.append(obj_array, f_k)
            opt_array = np.append(opt_array, np.linalg.norm(g_k))
            num_f_evals_array = np.append(
                num_f_evals_array, num_f_evals_array[-1] + num_f_evals)
            num_g_evals_array = np.append(
                num_g_evals_array, num_g_evals_array[-1] + num_g_evals)
            step_array = np.append(step_array, alpha)
            itr_end = time.time()
            time_array = np.append(
                time_array, [time_array[-1] + itr_end - itr_start])

            # Update output files with new values from the current iteration (passing the whole updated array rather than new values)
            # Note: We pass only x_k and not x_array (since x_array could be deprecated later)
            self.update_output_files(itr=itr_array,
                                     obj=obj_array,
                                     opt=opt_array,
                                     num_f_evals=num_f_evals_array,
                                     num_g_evals=num_g_evals_array,
                                     step=step_array,
                                     time=time_array,
                                     x=x_k)

        end_time = time.time()
        self.total_time = end_time - start_time

        # Update outputs_dict attribute at the end of optimization with the complete optimization history
        self.update_outputs_dict(itr=itr_array,
                                 x=x_array,
                                 obj=obj_array,
                                 opt=opt_array,
                                 num_f_evals=num_f_evals_array,
                                 num_g_evals=num_g_evals_array,
                                 step=step_array,
                                 time=time_array)
