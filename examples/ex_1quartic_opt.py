'''Minimizing a Quartic function'''

import numpy as np
from modopt import Problem


class X4(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'x^4'

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([.3, .3]))
        self.add_objective('f',)

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )
        self.declare_objective_hessian(of='x', wrt='x')

    # Compute the value of the objective, gradient and Hessian 
    # with the given design variable values
    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x']**3

    def compute_objective_hessian(self, dvs, hess):
        hess['x', 'x'] = 12 * np.diag(dvs['x']**2)

import time
from modopt import Optimizer

class SteepestDescent(Optimizer):
    def initialize(self):

        # Name your algorithm
        self.solver_name = 'steepest_descent'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('maxiter', default=1000, types=int)
        self.options.declare('opt_tol', types=float)
        # Enable user to specify, as a list, which among the available outputs
        # need to be written to output files
        self.options.declare('readable_outputs', types=list, default=[])

        # Specify format of outputs available from your optimizer after each iteration
        self.available_outputs = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
        }

    def solve(self):
        nx = self.problem.nx
        x = self.problem.x0
        opt_tol = self.options['opt_tol']
        maxiter = self.options['maxiter']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)

        # Iteration counter
        itr = 0

        # Optimality
        opt = np.linalg.norm(g_k)

        # Initializing outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time)

        while (opt > opt_tol and itr < maxiter):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>

            p_k = -g_k

            x_k += p_k
            f_k = obj(x_k)
            g_k = grad(x_k)

            opt = np.linalg.norm(g_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Append arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time

        self.results = {'x': x_k,
                        'objective': f_k,
                        'optimality': opt,
                        'niter': itr,
                        'time': self.total_time,
                        'converged': opt <= opt_tol}
        
        return self.results


# Set your optimality tolerance
opt_tol = 1E-8
# Set maximum optimizer iteration limit
maxiter = 100

prob = X4()

from modopt import Newton, QuasiNewton, SQP

# Set up your optimizer with your problem and pass in optimizer parameters
optimizer = SteepestDescent(prob,
                            opt_tol=opt_tol,
                            maxiter=maxiter,
                            readable_outputs=['itr', 'obj', 'x', 'opt', 'time'])
optimizer = Newton(prob, opt_tol=opt_tol)
optimizer = QuasiNewton(prob, opt_tol=opt_tol)

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()

# Print results of optimization (summary_table contains information from each iteration)
optimizer.print_results(summary_table=True)

# Print to see any output that was declared
# Since the arrays are long, here we only print the last entry and
# verify it with the print_results() above

print('\n')
print('Optimizer data')
print('num_iterations:', optimizer.results['niter'])
print('optimized_dvs:', optimizer.results['x'])
print('optimization_time:', optimizer.results['time'])
print('optimized_obj:', optimizer.results['objective'])
print('final_optimality:', optimizer.results['optimality'])

print('\n')
print('Final problem data')
print('optimized_dvs:', prob.x.get_data())
print('optimized_obj:', prob.obj['f'])