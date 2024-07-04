import numpy as np
import time

from modopt import Optimizer
from modopt.line_search_algorithms import ScipyLS, BacktrackingArmijo
from modopt.merit_functions import AugmentedLagrangianEq, L2Eq


class NelderMead(Optimizer):
    def initialize(self):
        self.solver_name = 'nelder_mead'

        self.obj = self.problem._compute_objective

        # Tolerance and initial length are empirical and problem-dependent
        self.options.declare('maxiter', default=200, types=int)
        self.options.declare('initial_length', default=1., types=float)
        self.options.declare('tol', default=1e-4, types=float)
        self.options.declare('outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'f_sd': float,
            'time': float,
            'num_f_evals': int,
        }

    def setup(self):
        self.setup_outputs()
        
    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        x0 = self.problem.x0
        tol = self.options['tol']
        maxiter = self.options['maxiter']
        l  = self.options['initial_length']
        obj = self.obj

        # x_k contains coordinates for all simplex vertices
        x_k = np.tile(x0, (nx+1,1))

        # Generate vertices for the initial simplex from initial x_0
        x_k[1:] += l/(np.sqrt(2)*nx) * (np.sqrt(nx+1)-1)
        x_k[1:] += l/np.sqrt(2) * np.identity(nx)

        start_time = time.time()

        # Iteration counter
        itr = 0

        f_k = np.zeros((nx+1,))
        for i in range(nx+1):
            f_k[i] = obj(x_k[i])

        f_sd = np.std(f_k)
        num_f_evals = nx + 1

        # # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k[np.argmin(f_k)],
                            obj=np.min(f_k),
                            f_sd=f_sd,
                            time=time.time() - start_time,
                            num_f_evals=num_f_evals,)

        while (f_sd > tol and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # SNAPSHOTS OF THE SIMPLEX CONVERGING
            # import matplotlib.pyplot as plt
            # plt.plot(x_k[:,0], x_k[:,1])
            # plt.scatter(x_k[:,0], x_k[:,1])
            # plt.xlim(-2,3)
            # plt.ylim(-1,3)
            # plt.show()
            # ######################################

            # Sort indices
            sorted_indices = np.argsort(f_k)
            f_k = f_k[sorted_indices]
            x_k = x_k[sorted_indices]
        
            # Centroid of all points except the worst
            x_c = np.sum(x_k[:-1], axis=0)/nx

            # Reflect the worst point about the centroid
            x_r = x_c + (x_c - x_k[-1])
            f_r = obj(x_r)
            num_f_evals += 1
            
            # When the reflection is better than the best
            if f_r < f_k[0]:
                # Try expansion along the reflected point
                x_e = x_c + 2 * (x_c - x_k[-1])
                f_e = obj(x_e)
                num_f_evals += 1

                if f_e < f_k[0]:
                    x_k[-1] = x_e
                    f_k[-1] = f_e
                else:
                    x_k[-1] = x_r
                    f_k[-1] = f_r

            # When the reflection is better than the second worst
            elif f_r <= f_k[-2]:
                x_k[-1] = x_r
                f_k[-1] = f_r

            else:
                # When the reflection is worse than the worst
                if f_r > f_k[-1]:
                    # Inner contraction
                    x_ic = x_c - 0.5 * (x_c - x_k[-1])
                    f_ic = obj(x_ic)
                    num_f_evals += 1

                    if f_ic < f_k[-1]:
                        x_k[-1] = x_ic
                        f_k[-1] = f_ic
                    else:   # Shrink       
                        for i in range(1, nx+1):
                            x_k[i] = x_k[0] + 0.5 * (x_k[i] - x_k[0])
                            f_k[i] = obj(x_k[i])
                            num_f_evals += 1

                # When the reflection is better than the worst
                # but worse than the second worst
                else:
                    # Outer contraction
                    x_oc = x_c + 0.5 * (x_c - x_k[-1])
                    f_oc = obj(x_oc)
                    num_f_evals += 1

                    if f_oc < f_r:
                        x_k[-1] = x_oc
                        f_k[-1] = f_oc
                    else:   # Shrink
                        for i in range(1, nx+1):
                            x_k[i] = x_k[0] + 0.5 * (x_k[i] - x_k[0])
                            f_k[i] = obj(x_k[i])
                            num_f_evals += 1

            f_sd = np.std(f_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k[np.argmin(f_k)],
                                obj=np.min(f_k),
                                f_sd=f_sd,
                                time=time.time() - start_time,
                                num_f_evals=num_f_evals,)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()
        self.total_time = time.time() - start_time
        converged = f_sd <= tol

        self.results = {
            'x': x_k[np.argmin(f_k)], 
            'f': np.min(f_k), 
            'f_sd': f_sd, 
            'nfev': num_f_evals,
            'niter': itr,
            'time': self.total_time,
            'converged': converged,
        }