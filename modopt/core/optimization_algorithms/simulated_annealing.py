import numpy as np
import time

from modopt.api import Optimizer

class SimulatedAnnealing(Optimizer):
    def initialize(self):
        self.solver_name = 'simulated_annealing'

        self.obj = self.problem._compute_objective
        self.neighbor = self.problem.generate_random_neighbor

        # emprirically chosen and problem dependent max_itr, settling time,
        # intial temperature, and tolerance criteria
        self.options.declare('max_itr', default=50000, types=int)
        self.options.declare('settling_time', default=100, types=int)
        self.options.declare('T0', default=1., types=float)
        self.options.declare('tol', default=1., types=float)

        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            'temp': float,
            # for arrays from each iteration, sizes need to be declared
            'x': (float, (self.problem.nx, )),
            'f_sd': float,
            'time': float,
        }
        self.options.declare('outputs',
                             types=list,
                             default=[
                                 'itr', 'obj', 'x', 'f_sd', 'time', 'temp',
                             ])

    def setup(self):
        pass
        
    def cool(self, T, itr):
        return 0.9 * T
        # # another possible cooling schedule
        # return T * (1-itr/self.options['max_itr'])

    def solve(self):
        # Assign shorter names to variables and methods
        nx = self.problem.nx
        x0 = self.problem.x.get_data().astype(int)
        tol = self.options['tol']
        max_itr = self.options['max_itr']
        T = self.options['T0']
        settling_time = self.options['settling_time']

        obj = self.obj
        neighbor = self.neighbor

        start_time = time.time()

        # Set intial values for current iterates
        x_k = x0 * 1
        f_k = obj(x_k)

        # Iteration counter
        itr = 0
        # Number of moves != itr
        num_moves = 0

        # Setting a high initial standard deviation
        f_sd = 100.

        # Allocating memory for latest 1000 objective values
        # for checking convergence. 1000 is an empirical number.
        # The convergence criteria is also empirical.
        f_hist = np.zeros((1000,))
        f_hist[0] = f_k

        # Initializing declared outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            temp=T,
                            f_sd=f_sd,
                            time=time.time() - start_time,
                            )

        while (f_sd > tol and itr < max_itr):
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

            if f_new <= f_k:
                x_k[:] = x_new
                f_k    = f_new
                num_moves += 1
                f_hist[num_moves%1000] = f_new
            else:
                r = np.random.rand()
                # probability of accepting a worse point
                p = np.exp((f_k-f_new)/T)
                if r <= p:
                    x_k[:] = x_new
                    f_k    = f_new
                    num_moves += 1
                    f_hist[num_moves%1000] = f_new

            # compute convergence criterion
            # standard deviation of latest 1000 objective values
            if num_moves >= 1000:
                f_sd = np.std(f_hist)

            print(f"{itr}: {f_k}")

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Update arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                f_sd=f_sd,
                                temp=T,
                                time=time.time() - start_time,)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time