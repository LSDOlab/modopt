'''Minimizing the Bean function'''

import numpy as np
from modopt import Problem

class BeanFunction(Problem):
    def initialize(self, ):
        self.problem_name = 'bean_function'

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                #   vals=np.array([-1,1.5]),
                                  vals=np.zeros(2,),
                                  lower=-10*np.ones(2,),
                                  upper=10*np.ones(2,))
        self.add_objective('f')


    def setup_derivatives(self):
        self.declare_objective_gradient(wrt='x', vals=None)

    def compute_objective(self, dvs, obj):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        obj['f'] = (1-x1)**2 + (1-x2)**2 + 0.5*(2*x2 - x1**2)**2

    def compute_objective_gradient(self, dvs, grad):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        dfdx1 = 2 * (1-x1) - 2*x1 * (2*x2 - x1**2)
        dfdx2 = 2 * (1-x2) + 2 * (2*x2 - x1**2)
        grad['x'] = np.array([dfdx1, dfdx2])

from modopt import PSO, NelderMead

prob = BeanFunction()

# Empirically arrived best hyperparameter values for BeanFunction for tol=1e-4:
# population=20, w = 0.5, c_g = 0.3, c_p = 0.4
tol = 1e-4
max_itr = 500
population = 20
w = 0.5
c_p = 0.4
c_g = 0.3

# Set up your PSO optimizer with the problem
optimizer = PSO(prob, 
                max_itr=max_itr, 
                tol=tol, 
                population=population,
                inertia_weight=w,
                cognitive_coeff=c_p,
                social_coeff=c_g,)

optimizer.solve()
optimizer.print_results(summary_table=True)

# import matplotlib.pyplot as plt
# y = np.loadtxt("bean_function_outputs/obj.out")
# plt.plot(y)
# plt.show()

print('PSO results:')

print('optimized_dvs:', optimizer.results['x'])
print('optimized_obj:', optimizer.results['f'])
print('final population obj std dev:', optimizer.results['f_sd'])
print('total number of function evaluations:', optimizer.results['nfev'])
print('total number of iterations:', optimizer.results['niter'])
print('total time taken:', optimizer.results['time'])
print('converged:', optimizer.results['converged'])


# Set up your Nelder-Mead optimizer with the problem
prob = BeanFunction()
tol = 1e-6
max_itr = 200
l = 1.0
optimizer = NelderMead(prob, max_itr=max_itr, tol=tol, initial_length=l)

optimizer.solve()
optimizer.print_results(summary_table=True)

# import matplotlib.pyplot as plt
# y = np.loadtxt("bean_function_outputs/obj.out")
# plt.plot(y)
# plt.show()

print('Nelder-Mead results:')

print('optimized_dvs:', optimizer.results['x'])
print('optimized_obj:', optimizer.results['f'])
print('final population obj std dev:', optimizer.results['f_sd'])
print('total number of function evaluations:', optimizer.results['nfev'])
print('total number of iterations:', optimizer.results['niter'])
print('total time taken:', optimizer.results['time'])
print('converged:', optimizer.results['converged'])