import numpy as np
import time
from modopt import Optimizer

class SteepestDescentNoLS(Optimizer):
    def initialize(self):
        self.solver_name = 'steepest_descent-no_ls'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('maxiter', types=int, default=500)
        self.options.declare('opt_tol', types=float, default=1e-5)
        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
            'nfev': int,
            'ngev': int,
        }

    def setup(self):
        pass

    def solve(self):

        # Assign shorter names to variables and methods
        nx = self.problem.nx
        x0 = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Iteration counter
        itr = 0

        opt = np.linalg.norm(g_k)
        nfev = 1
        ngev = 1

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time,
                            nfev=nfev,
                            ngev=ngev,)

        while (opt > opt_tol and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Compute the search direction toward the next iterate as the negative of the gradient
            p_k = -g_k

            x_k += p_k * 1.
            f_k = obj(x_k)
            g_k = grad(x_k)

            nfev += 1
            ngev += 1

            opt = np.linalg.norm(g_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time,
                                nfev=nfev,
                                ngev=ngev,)

        self.total_time = time.time() - start_time
        converged = opt <= opt_tol

        self.results = {
            'x': x_k, 
            'objective': f_k, 
            'optimality': opt, 
            'nfev': nfev, 
            'ngev': ngev,
            'niter': itr, 
            'time': self.total_time,
            'converged': converged,
            }
        
        # Run post-processing for the Optimizer() base class
        self.run_post_processing()
        
        return self.results