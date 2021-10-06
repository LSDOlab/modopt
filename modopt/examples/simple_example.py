import numpy as np

from modopt.api import Problem


class X2(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'x^4'

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([5., 10.]))

        # # Add the objective your problem
        # self.add_objective('obj')

    def setup_derivatives(self):
        # Declare objective gradient and it's shape
        self.declare_objective_gradient(
            wrt='x',
            shape=(2, ),
        )

    # Compute the value of the objective with given design variable values
    def compute_objective(self, x):
        return x[0]**2 + x[1]**2

    def compute_objective_gradient(self, x):
        return 2 * x


import numpy as np
import time

from modopt.api import Optimizer


class SteepestDescent(Optimizer):
    def initialize(self):
        # Name your algorithm
        self.solver = 'steepest_descent'

        self.obj = self.problem.compute_objective
        self.grad = self.problem.compute_objective_gradient

        self.options.declare('opt_tol', types=float)

    def solve(self):
        nx = self.problem.nx
        x0 = x0 = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        max_itr = self.options['max_itr']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Setting intial values for current iterates
        x_k = x0 * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Setting intial values for computed new iterates
        f_new = f_k * 1
        g_new = g_k * 1

        itr = 0

        # Initializing output arrays
        itr_array = np.array([
            0,
        ])
        x_array = x0.reshape(1, nx)
        obj_array = np.array([f_k * 1.])
        opt_array = np.array([np.linalg.norm(g_k)])

        time_array = np.array([time.time() - start_time])

        while (opt_array[-1] > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            p_k = -g_k

            x_k += p_k
            f_k = obj(x_k)
            g_k = grad(x_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Append output arrays with new values from the current iteration
            itr_array = np.append(itr_array, itr)
            x_array = np.append(x_array, x_k.reshape(1, nx), axis=0)
            obj_array = np.append(obj_array, f_k)
            opt_array = np.append(opt_array, np.linalg.norm(g_k))
            itr_end = time.time()
            time_array = np.append(
                time_array, [time_array[-1] + itr_end - itr_start])

            # Update output files with new values from the current iteration (passing the whole updated array rather than new values)
            # Note: We pass only x_k and not x_array (since x_array could be deprecated later)
            self.update_output_files(itr=itr_array,
                                     obj=obj_array,
                                     opt=opt_array,
                                     time=time_array,
                                     x=x_k)

        end_time = time.time()
        self.total_time = end_time - start_time

        # Update outputs_dict attribute at the end of optimization with the complete optimization history
        self.update_outputs_dict(itr=itr_array,
                                 x=x_array,
                                 obj=obj_array,
                                 opt=opt_array,
                                 time=time_array)


# Set your optimality tolerance
opt_tol = 1E-8
# Set maximum optimizer iteration limit
max_itr = 500

prob = X2()

# Set up your optimizer with your problem and pass in optimizer parameters
optimizer = SteepestDescent(prob, opt_tol=opt_tol, max_itr=max_itr)

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x.get_data())

# Solve your optimization problem
optimizer.solve()

# Print results of optimization (summary_table contains information from each iteration)
optimizer.print_results(summary_table=True)