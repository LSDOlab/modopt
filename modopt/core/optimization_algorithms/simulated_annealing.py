import numpy as np
import time

from modopt import Optimizer

class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing algorithm for discrete, unconstrained optimization.

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

    maxiter : int, default=50000
        Maximum number of iterations.
    settling_time : int, default=100
        Number of iterations after which the temperature is decreased.
    T0 : float, default=1.
        Initial temperature.
    std_dev_tol : float, default=1.
        Tolerance for the convergence criterion.
        Certifies convergence when the standard deviation of the latest
        `std_dev_sample_size` function values is less than this value.
    std_dev_sample_size : int, default=1000
        Number of latest function values to consider for computing the
        standard deviation for checking the convergence criterion.
    readable_outputs : list, default=[]
        List of outputs to be written to readable text output files.
        Available outputs are: 'itr', 'obj', 'temp', 'x', 'f_sd', and 'time'.
    """
    def initialize(self):
        self.solver_name = 'simulated_annealing'

        self.obj = self.problem._compute_objective
        self.neighbor = self.problem.get_neighbor

        # emprirically chosen and problem dependent maxiter, settling time,
        # intial temperature, and tolerance criteria
        self.options.declare('maxiter', default=50000, types=int)
        self.options.declare('settling_time', default=100, types=int)
        self.options.declare('T0', default=1., types=float)
        self.options.declare('std_dev_tol', default=1., types=float)
        self.options.declare('std_dev_sample_size', default=1000, types=int)
        self.options.declare('readable_outputs', types=list, default=[])

        self.available_outputs = {
            'itr': int,
            'obj': float,
            'temp': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'f_sd': float,
            'time': float,
        }
    
    def setup(self):
        pass

    def cool(self, T, itr):
        return 0.9 * T
        # # another possible cooling schedule
        # return T * (1-itr/self.options['maxiter'])

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        x0 = self.problem.x0.astype(int)
        tol = self.options['std_dev_tol']
        hist_size = self.options['std_dev_sample_size']
        maxiter = self.options['maxiter']
        T = self.options['T0']
        settling_time = self.options['settling_time']

        obj = self.obj
        neighbor = self.neighbor

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1
        f_k = obj(x_k)
        f0 = f_k * 1

        # Set best values found so far as the initial values
        x_best = x_k * 1
        f_best = f_k * 1

        # Iteration counter
        itr = 0
        # Number of moves != itr
        num_moves = 0

        # Setting a high initial standard deviation
        f_sd = 100.

        # Allocating memory for latest 1000 objective values
        # for checking convergence. 1000 is an empirical number.
        # The convergence criteria is also empirical.
        f_hist = np.zeros((hist_size,))
        f_hist[0] = f_k

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            temp=T,
                            f_sd=f_sd,
                            time=time.time() - start_time,
                            )

        while (f_sd > tol and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            # Decrease temperature according to some 
            # cooling schedule after settling time
            if itr%settling_time == 0:
                T = self.cool(T, itr)

            # Neighbor generation
            x_new = neighbor(x_k)
            f_new = obj(x_new)
            # f_hist[itr%1000] = f_new

            # Update the best solution found so far
            if f_new < f_best:
                x_best = x_new * 1
                f_best = f_new * 1

            if f_new <= f_k:
                x_k[:] = x_new
                f_k    = f_new
                num_moves += 1
                # f_hist[num_moves%1000] = f_new
            else:
                r = np.random.rand()
                # probability of accepting a worse point
                p = np.exp((f_k-f_new)/T)
                if r <= p:
                    x_k[:] = x_new
                    f_k    = f_new
                    num_moves += 1
                    # f_hist[num_moves%1000] = f_new
            f_hist[itr%hist_size] = f_k

            # compute convergence criterion
            # standard deviation of latest `hist_size` objective values
            if num_moves >= hist_size:
                f_sd = np.std(f_hist)

            # print(f"{itr}: {f_k}")

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                f_sd=f_sd,
                                temp=T,
                                time=time.time() - start_time,)

        self.total_time = time.time() - start_time
        converged = f_sd <= tol
        improvement = (f0 - f_best)/f0

        self.results = {
            'x': x_best,
            'objective': f_best,
            'f0': f0,
            'f_sd': f_sd,
            'nfev': itr+1,
            'niter': itr,
            'time': self.total_time,
            'converged': converged,
            'nmoves': num_moves,
            'improvement': improvement,
        }

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        return self.results