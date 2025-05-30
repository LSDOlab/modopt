import numpy as np
import time
from scipy.stats import qmc

from modopt import Optimizer
from modopt.merit_functions import AugmentedLagrangianEq, L2Eq
import warnings


class PSO(Optimizer):
    """
    The Particle Swarm Optimization (PSO) algorithm for unconstrained optimization
    within a bounded domain.

    Note that the problem needs to define the bounds for each variable.
    Otherwise, the default bounds of -10 and +10 are used for unbounded variables.

    Parameters
    ----------
    problem : Problem or ProblemLite
        Object containing the problem to be solved.
    recording : bool, default=False
        If ``True``, record all outputs from the optimization.
        This needs to be enabled for hot-starting the same problem later,
        if the optimization is interrupted.
    hot_start_from : str, optional
        The record file from which to hot-start the optimization.
    hot_start_atol : float, default=0.
        The absolute tolerance check for the inputs 
        when reusing outputs from the hot-start record.
    hot_start_rtol : float, default=0.
        The relative tolerance check for the inputs 
        when reusing outputs from the hot-start record.
    visualize : list, default=[]
        The list of scalar variables to visualize during the optimization.
    keep_viz_open : bool, default=False
        If ``True``, keep the visualization window open after the optimization is complete.
    turn_off_outputs : bool, default=False
        If ``True``, prevent modOpt from generating any output files.

    population : int, default=25
        Number of particles in the swarm.
    maxiter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for the convergence criterion.
        Certifies convergence when the standard deviation of the 
        objective function values of the particles is less than this value.
    inertia_weight : float, default=0.8
        Inertia weight for the velocity update.
    cognitive_coeff : float, default=0.1
        Cognitive coefficient (self influence) for the velocity update.
    social_coeff : float, default=0.1
        Social coefficient (social influence) for the velocity update.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'itr', 'obj', 'x', 'f_sd', 'time', and 'nfev'.
    """
    def initialize(self):
        self.solver_name = 'pso'

        self.obj = self.problem._compute_objective

        # All the options below are empirically chosen for a problem
        self.options.declare('population', default=25, types=int)
        self.options.declare('maxiter', default=100, types=int)
        self.options.declare('tol', default=1e-4, types=float)
        self.options.declare('inertia_weight', default=0.8, types=float)
        self.options.declare('cognitive_coeff', default=0.1, types=float)
        self.options.declare('social_coeff', default=0.1, types=float)
        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'f_sd': float,
            'time': float,
            'nfev': int,
        }

    def setup(self):
        pass

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        # x0 = self.problem.x0
        tol = self.options['tol']
        population = self.options['population']
        maxiter = self.options['maxiter']
        w   = self.options['inertia_weight']
        c_p = self.options['cognitive_coeff']
        c_g = self.options['social_coeff']

        obj = self.obj

        # Generate initial particle locations and velocities using
        # Latin Hypercube Sampling between variable bounds
        sampler = qmc.LatinHypercube(d=nx, seed=0)
        sample = sampler.random(n=population)
        x_min = self.problem.x_lower
        x_max = self.problem.x_upper

        if any(x_min == -np.inf) or any(x_max == np.inf):
            warnings.warn('Infinite variable bounds detected. Using default bounds of -10 and +10 for infinite bounds.')
            x_min = np.where(x_min == -np.inf, -10, x_min)
            x_max = np.where(x_max ==  np.inf,  10, x_max)
        
        x0 = qmc.scale(sample, x_min, x_max)

        sampler = qmc.LatinHypercube(d=nx, seed=1)
        sample = sampler.random(n=population)
        # Limit the velocity to 5 percent of the diffence between upper and lower bounds
        v_max = 0.05 * (x_max - x_min)
        v_min = -v_max
        v0 = qmc.scale(sample, v_min, v_max)

        start_time = time.time()

        # Set initial values for current iterates
        x_k = x0 * 1.
        v_k = v0 * 1.
        f_k = np.full((population,), np.inf)

        # Setting a high initial standard deviation
        f_sd = 100.

        x_best_p = x_k * 1.
        f_best_p = np.full((population,), np.inf)
        f_best_g = np.inf

        # Iteration counter
        itr = 0
        nfev = 0

        # Seed should be removed if benchmarking against other algorithms
        # with many runs of the same problem from random initializations
        np.random.seed(0)

        while (f_sd > tol and itr < maxiter):
            itr_start = time.time()

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            for i in range(population):
                f_p = obj(x_k[i])
                f_k[i] = f_p

                if f_p < f_best_p[i]:
                     x_best_p[i] = x_k[i]
                     f_best_p[i] = f_p
                     if f_p < f_best_g:
                        x_best_g = x_k[i]
                        f_best_g = f_p

            for i in range(population):
                r_p = np.random.rand()
                r_g = np.random.rand()
                v_p = w * v_k[i] + c_p*r_p * (x_best_p[i] - x_k[i]) + c_g*r_g * (x_best_g - x_k[i])
                v_k[i] = np.clip(v_p, v_min, v_max)
                x_p = x_k[i] + v_k[i]
                x_k[i] = np.clip(x_p, x_min, x_max)

            nfev += population
            f_sd = np.std(f_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # SNAPSHOTS OF SWARM CONVERGING
            # TODO: Show velocity vectors also for each particle
            # import matplotlib.pyplot as plt
            # plt.scatter(x_k[:,0], x_k[:,1])
            # plt.xlim(-10,10)
            # plt.ylim(-10,10)
            # plt.show()

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_best_g,
                                obj=f_best_g,
                                # obj=np.min(f_k),
                                f_sd=f_sd,
                                time=time.time() - start_time,
                                nfev=nfev,)
            
            itr += 1
            
        self.total_time = time.time() - start_time
        converged = f_sd <= tol

        self.results = {
            'x': x_best_g, 
            'objective': f_best_g, 
            'f_sd': f_sd, 
            'nfev': nfev, 
            'niter': itr-1, 
            'time': self.total_time,
            'converged': converged,
            }
        
        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results