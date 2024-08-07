'''Benchmark performant algorithms on three analytical problems'''

import numpy as np
from modopt import ProblemLite, optimize
from modopt.postprocessing import load_variables
import time
import contextlib
import io
import matplotlib.pyplot as plt

# Problem 1: minimize sum(x^2)
x0 = np.array([1., 1.])
name = 'quartic_function'
obj  = lambda x: np.sum(x**4)
grad = lambda x: 4 * x**3
obj_hess = lambda x: 12 * np.diag(x**2)
prob1 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol1  = np.array([0., 0.])

# Problem 2: minimize (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
x0 = np.array([-1.2, 1.])
name = 'rosenbrock_function'
obj  = lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
grad = lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
obj_hess = lambda x: np.array([[2 + 800*x[0]**2 - 400*(x[1]-x[0]**2), -400*x[0]], [-400*x[0], 200]])
prob2 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol2  = np.array([1., 1.])

# Problem 3: minimize (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2
x0 = np.array([0., 0.])
name = 'bean_function'
obj  = lambda x: (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1] - x[0]**2)**2
grad = lambda x: np.array([-2*(1-x[0]) - 2*x[0]*(2*x[1] - x[0]**2), -2*(1-x[1]) + 2*(2*x[1] - x[0]**2)])
obj_hess = lambda x: np.array([[2 + 4*x[0]**2 - 2*(2*x[1] - x[0]**2), -4*x[0]], [-4*x[0], 2 + 4]])
prob3 = ProblemLite(name=name, x0=x0, obj=obj, grad=grad, obj_hess=obj_hess)
sol3  = np.array([1.21314, 0.82414])

# Benchmarking optimization algorithms 
# NOTE: TrustConstr will use obj_hess by default while IPOPT will not
algs = ['SNOPT', 'IPOPT', 'IPOPT-2', 'PySLSQP', 'BFGS', 'LBFGSB', 
        'COBYLA', 'COBYQA', 'NelderMead', 'TrustConstr', 'TrustConstr-2']

performance = {}
history = {}
time_loop = 20

probs = [prob1, prob2, prob3]
sols  = [sol1, sol2, sol3]

for prob, sol in zip(probs, sols):
    print('\nProblem:', prob.problem_name)
    print('='*50)

    for alg in algs:
        solver = alg
        options = {}
        if alg=='IPOPT-2':
            solver = 'IPOPT'
            options = {'hessian_approximation': 'exact'}
        elif alg=='TrustConstr-2':
            solver = 'TrustConstr'
        elif alg == 'TrustConstr':
            options = {'ignore_exact_hessian': True}
    
        print(f'\t{alg} \n\t------------------------')
        start_time = time.time()
        for i in range(time_loop-1):
            with contextlib.redirect_stdout(io.StringIO()):
                results = optimize(prob, solver=solver, solver_options=options,  recording=False, turn_off_outputs=True)
        results  = optimize(prob, solver=solver, solver_options=options,  recording=True)
        opt_time = (time.time() - start_time) / time_loop
        success  = np.allclose(results['x'], sol, atol=1e-3)

        nev       = prob._callback_count
        o_evals   = prob._obj_count
        g_evals   = prob._grad_count
        h_evals   = prob._hess_count
        objective = prob._compute_objective(results['x'])
        print('\tTime:', opt_time)
        print('\tSuccess:', success)
        print('\tEvaluations:', nev)
        print('\tObj evals:', o_evals)
        print('\tGrad evals:', g_evals)
        print('\tHess evals:', h_evals)
        print('\tOptimized vars:', results['x'])

        obj_hist = load_variables(f"{results['out_dir']}/record.hdf5", 'obj')['callback_obj']
        history[prob.problem_name, alg] = obj_hist

        performance[prob.problem_name, alg] = {'time': opt_time,
                                               'success': success,
                                               'nev': nev,
                                               'objective': objective}
        
    plt.figure()
    for alg in algs:
        y_data = history[prob.problem_name, alg]
        plt.semilogy(y_data, label=f"{alg} ({len(y_data)})")
    plt.xlabel('Evaluations')
    plt.ylabel('Objective')
    plt.title(f'{prob.problem_name} minimization')
    plt.legend()
    plt.grid()
    plt.savefig(f"{prob.problem_name}-objective-cb.pdf")
    plt.close()


# Print performance
print('\nPerformance')
print('='*50)
for key, value in performance.items():
    print(f"{str(key):45}:", value)

from modopt.benchmarking import plot_performance_profiles
plot_performance_profiles(performance, save_figname='performance.pdf')