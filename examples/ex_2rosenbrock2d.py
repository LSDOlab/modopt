'''Minimizing the Rosenbrock function'''

import numpy as np
from modopt import Problem


class Rosenbrock2D(Problem):

    def initialize(self):
        # Name your problem
        self.problem_name = 'rosenbrock'

    def setup(self):
        # Add the design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=None,
                                  upper=None,
                                  vals=np.array([-1.2, 1.]))

        self.add_objective('f')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )
        self.declare_objective_hessian(of='x', wrt='x')

    # Compute the value of the objective, gradient and Hessian 
    # with the given design variable values
    def compute_objective(self, dvs, obj):
        x = dvs['x']
        obj['f'] = (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def compute_objective_gradient(self, dvs, grad):
        x = dvs['x']
        grad['x'] = np.array([
            -400 * x[0] * (x[1] - x[0]**2) + 2 * (x[0] - 1),
            200 * (x[1] - x[0]**2)
        ])

    def compute_objective_hessian(self, dvs, hess):
        x = dvs['x']
        hess['x', 'x'] = np.array([
            [2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
            [-400 * x[0], 200]
            ])
        

if __name__ == "__main__":
    from modopt import Newton, QuasiNewton, SteepestDescent

    # Set your optimality tolerance
    opt_tol = 1E-8
    # Set maximum optimizer iteration limit
    maxiter = 100

    prob = Rosenbrock2D()

    # Set up your optimizer with your problem and pass in optimizer parameters
    # optimizer = SteepestDescent(prob,
    #                             opt_tol=opt_tol,
    #                             maxiter=maxiter,
    #                             outputs=['itr', 'obj', 'x', 'opt', 'time'])
    # optimizer = Newton(prob, opt_tol=opt_tol)
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
    print('optimized_obj:', optimizer.results['f'])
    print('final_optimality:', optimizer.results['optimality'])

    print('\n')
    print('Final problem data')
    print('optimized_dvs:', prob.x.get_data())
    print('optimized_obj:', prob.obj['f'])